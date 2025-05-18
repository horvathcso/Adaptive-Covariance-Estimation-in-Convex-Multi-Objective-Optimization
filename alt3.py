# aco_active_learning.py
import pandas as pd
import numpy as np
from collections import defaultdict
from surrogate_model import fit_gp_model, optimize_for_lambda, hessian_estimation_for_lambda, estimate_local_covariances_from_lambdas,load_lambda_covariance_data,train_and_prepare_surrogate
import matplotlib.pyplot as plt
import os
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
import concurrent.futures
from filelock import FileLock
import time

NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations
GLOBAL_SEED = 42 # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)

class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None, rho=0.1):  # aggiungi rho
        self.lambda_data = lambda_data.copy()
        self.lambda_data['lambda3'] = 1 - self.lambda_data['lambda1'] - self.lambda_data['lambda2']
        self.lambdas = self.lambda_data[['lambda1', 'lambda2', 'lambda3']].values

        self.tau = defaultdict(lambda: 1.0)
        self.eta = {} # Heuristic values, norma
        self.surrogate = surrogate_model

        self.Selected = set()
        self.C_e = None

        self.epsilon_threshold = 1e-2
        self.rho = rho  # salva rho
    # Calclo della norma di sensibilità
    def compute_heuristic(self):
        if self.surrogate is None:
            raise ValueError("Surrogate model non definito.")
        # Calcola la norma di sensibilità per ogni lambda , deve usare il modello surrogato
        print("Calcolo eucristica per ogni lambda...")
        X = self.lambda_data[['lambda1', 'lambda2']].values
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        for i, lam in enumerate(self.lambdas):
            self.eta[tuple(lam)] = y_pred[i]
            

    # Restituisce i valori di tau e eta
    def get_pheromone_and_heuristic(self):
        return self.tau, self.eta

    def get_initial_state(self):
        return self.Selected, self.C_e, self.epsilon_threshold


    def select_initial_diverse_lambdas(self, n_init_diverse=5, random_state=None, exclude_lambdas=None):
        """
        Seleziona n_init_diverse lambda iniziali massimizzando la diversità (distanza euclidea).
        """
        if exclude_lambdas is None:
            exclude_lambdas = set()
        if not self.eta:
            raise ValueError("Euristica (eta) non calcolata. Chiama compute_heuristic() prima.")

        sorted_lambdas = sorted(self.eta.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(sorted_lambdas)

        selected = []
        seen = set()
        for lam, score in sorted_lambdas:
            if lam not in seen and lam not in exclude_lambdas:
                selected.append(np.array(lam))
                seen.add(lam)
            if len(selected) >= n_init_diverse:
                break

        for lam in selected:
            self.Selected.add(tuple(lam))
        print(f"Lambda iniziali selezionati (norma massima): {selected}")

    # campionamento dei valori di lamnda da assegnare alle formiche in base a tau e eta
    def sample_candidates(self, n_ants=20, alpha=1.0, beta=1.0):
           
        print("Campionamento candidati...")
        keys = list(self.eta.keys())
        scores = np.array([
            (self.tau[lam] ** alpha) * (self.eta[lam] ** beta) for lam in keys
        ])
        # Rendi i punteggi non negativi
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-8
        if np.sum(scores) == 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / np.sum(scores)
        print(f"Probabilità di campionamento: {probs}")
        indices = np.random.choice(len(keys), size=n_ants, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        print(f"Candidati campionati: {sampled}")
        return [keys[i] for i in indices]

    # Aggiorna la lista di lambda selezionati e calcola la matrice di covarianza empirica C_e
    def update_selected_and_Ce(self, selected_lambda, C_lambda=None):
        self.Selected.add(tuple(selected_lambda))
        print("Lambda attualmente in Selected:")
        for lam in self.Selected:
            print(lam)
        if len(self.Selected) <= 1:
            print("Numero insufficiente di elementi selezionati per calcolare C_e.")
            self.C_e = None
            return

        x_list = []
        for lam in self.Selected:
            _, x_opt = optimize_for_lambda(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar

        # Calcola C_e solo se ci sono abbastanza elementi
        if len(self.Selected) > 1:
            self.C_e = (deviations.T @ deviations) / (len(self.Selected) - 1)
            print(f"Dimensione di C_e: {self.C_e}")
        else:
            self.C_e = None
       

    def run_aco_active_learning(self, C_ref, archive_data, n_ants=50, top_k=20,
                                alpha=1.0, beta=1.0, omega=0.7,
                                epsilon=None, budget=30, retrain_every=6,
                                n_init_diverse=5, exclude_lambdas=None, random_state=None):
        if epsilon is None:
            epsilon = self.epsilon_threshold

        # Selezione iniziale diversificata SOLO se Selected è vuoto
        if len(self.Selected) == 0:
            self.select_initial_diverse_lambdas(
                n_init_diverse=n_init_diverse,
                random_state=random_state,
                exclude_lambdas=exclude_lambdas
            )
            self.update_selected_and_Ce(next(iter(self.Selected)))

        val_count = len(self.Selected)
        error_list = []
        while val_count < budget:
            print(f"Iterazione {val_count + 1}/{budget}", flush=True)
            candidates = self.sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)

            for lam in candidates:
                X = np.array([[lam[0], lam[1]]])
                X_scaled = self.surrogate['scaler_X'].transform(X)
                y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
                y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                self.eta[lam] = y_pred[0]
                

            delta_k_list = []
            for lam in candidates:
                delta_k = self.compute_delta_k(C_ref, lam)
                delta_k_list.append((lam, delta_k))
                
            eta_vals = np.array([self.eta[lam] for lam, _ in delta_k_list])
            delta_vals = np.array([abs(dk) for _, dk in delta_k_list])

            # Normalizza su [0, 1]
            eta_norm = (eta_vals - eta_vals.min()) / (eta_vals.ptp() + 1e-8)
            delta_norm = (delta_vals - delta_vals.min()) / (delta_vals.ptp() + 1e-8)


            scored_candidates = []
            for i, (lam, _) in enumerate(delta_k_list):
                score = omega * eta_norm[i] - (1 - omega) * delta_norm[i]
                scored_candidates.append((lam, score))

            scored_candidates.sort(key=lambda x: x[1])
            top_candidates = [lam for lam, _ in scored_candidates[:top_k]]
            print(f"Top-k candidati: {top_candidates}")

            # --- EVAPORAZIONE DEL FEROMONE ---
            for key in self.tau:
                self.tau[key] = (1 - self.rho) * self.tau[key]

            batch_xopt = []
            batch_cov = []
            batch_sens = []

            # Valuta tutti i top-k in parallelo
            for lam in top_candidates:
                _, x_opt = optimize_for_lambda(lam)
                x_opt_flat = np.asarray(x_opt).flatten()
                x_list = []
                for l in self.Selected:
                    _, x_old = optimize_for_lambda(l)
                    x_old_flat = np.asarray(x_old).flatten()
                    x_list.append(x_old_flat)
                x_list.append(x_opt_flat)
                X = np.vstack(x_list)
                x_bar = np.mean(X, axis=0)
                deviations = X - x_bar
                if len(x_list) > 1:
                    cov_matrix = (deviations.T @ deviations) / (len(x_list) - 1)
                else:
                    cov_matrix = np.zeros((X.shape[1], X.shape[1]))
                sens_norm = np.linalg.norm(cov_matrix, ord='fro')
                batch_xopt.append(x_opt)
                batch_cov.append(cov_matrix)
                batch_sens.append(sens_norm)
                # Aggiorna feromone e eta
                self.tau[lam] += sens_norm
                self.eta[lam] = sens_norm


            # Aggiorna Selected con tutto il batch, evitando duplicati
            new_lambdas = []
            for lam in top_candidates:
                if tuple(lam) not in self.Selected:
                    self.Selected.add(tuple(lam))
                    new_lambdas.append(lam)
                    print(f"Nuovo lambda selezionato: {lam}")
                else:
                    print(f"ATTENZIONE: Lambda già presente in Selected, salto: {lam}")
# Compute the covariance matrix and optimal solution for the given lambda

            
            # Aggiorna C_e una sola volta
            self.update_selected_and_Ce(top_candidates[0])  # aggiorna su tutto Selected

            # Calcola errore solo una volta
            if self.C_e is not None:
                error = np.linalg.norm(C_ref - self.C_e, ord='fro')
            else:
                error = None

            # Stampa batch
            print(f"Batch selezionato: {top_candidates}")
            for lam, sens in zip(top_candidates, batch_sens):
                print(f"  Lambda: {lam}, Sensitivity Norm: {sens:.2f}")
            if error is not None:
                print(f"Errore attuale dopo batch: {error:.4f}")
            else:
                print("Errore attuale: N/A (C_e non definita)")

            val_count += len(top_candidates)
            error_list.append(error)

            '''if val_count - len(top_candidates) < retrain_every or (val_count // retrain_every > (val_count - len(top_candidates)) // retrain_every):

                print("Riaddestramento del modello surrogato...")
                self.retrain_surrogate(archive_data)

            self.surrogate, self.last_mtime = maybe_reload_surrogate(
                self.surrogate, model_path="surrogate_global.pkl", last_mtime=getattr(self, 'last_mtime', None)
            )'''

        plt.figure(figsize=(8,4))
        plt.plot(error_list, marker='o')
        plt.title("Errore Frobenius tra $C_{ref}$ e $C_e$ per iterazione")
        plt.xlabel("Iterazione")
        plt.ylabel("Errore Frobenius")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("errore_per_iterazione.png")
        plt.show()

    def compute_Ce_from_lambdas(self, lambdas_list):
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            _, x_opt = optimize_for_lambda(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        C_e = (deviations.T @ deviations) / (len(lambdas_list) - 1)
        return C_e

    # Calcola la variazione dell'errore dall'aggiunta del nuovo lambda
    def compute_delta_k(self, C_ref, lam):
        if not self.Selected:
            C_e_single = self.compute_Ce_from_lambdas([lam])
            delta_k = -np.linalg.norm(C_ref - C_e_single, ord='fro')
        else:
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected) + [lam])
            delta_k = np.linalg.norm(C_ref - C_e_new, ord='fro') - np.linalg.norm(C_ref - C_e_old, ord='fro')
        return delta_k

    def retrain_surrogate(self, archive_data):
        print("Riaddestro il modello surrogato con i seguenti nuovi lambda:")
        for lam in self.Selected:
            print(f"  {lam}")
        # Riaddestramento del modello surrogato
        selected_records = []
        for lam in self.Selected:
            # Calcola x_opt e la matrice di covarianza
            cov_matrix, x_opt = hessian_estimation_for_lambda(lam)

            # Calcola Sigma_x e Sigma_f
            Sigma_x, Sigma_f = estimate_local_covariances_from_lambdas(lambda_vec=lam,num_perturbations=NUM_PERTURBATIONS,delta=PERTURBATION_STRENGTH)

            # Calcola la matrice triangolare P e la sua versione appiattita
            try:
                P = np.linalg.cholesky(cov_matrix).T
                P_flattened = P[np.triu_indices_from(P)]
            except np.linalg.LinAlgError:
                P = np.full_like(cov_matrix, np.nan)
                P_flattened = np.full((cov_matrix.shape[0] * (cov_matrix.shape[0] + 1)) // 2, np.nan)

            # Registra i dati calcolati
            record = {
                'lambda1': lam[0],
                'lambda2': lam[1],
                'lambda3': lam[2],
                'x_opt': x_opt.tolist(),
                'cov_matrix': cov_matrix.tolist(),
                'solution_covariance': Sigma_x.tolist(),
                'objective_covariance': Sigma_f.tolist(),
                'sensitivity_norm': np.linalg.norm(cov_matrix, ord='fro'),
                'P_matrix': P.tolist(),
                'P_flattened': P_flattened.tolist()
            }
            selected_records.append(record)

            # Combina i dati selezionati con quelli di archivio
        selected_df = pd.DataFrame(selected_records)
        combined_df = pd.concat([archive_data, selected_df], ignore_index=True)

        # Riallena il modello surrogato
        model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = fit_gp_model(combined_df, n_training=len(combined_df))
        self.surrogate = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        X_eval = scaler_X.transform(combined_df[['lambda1', 'lambda2']].values)
        y_true = combined_df['sensitivity_norm'].values
        y_scaled = scaler_y.transform(y_true.reshape(-1, 1)).ravel()
        y_pred_scaled = model.predict(X_eval)
        mse = mean_squared_error(y_scaled, y_pred_scaled)
        r2 = r2_score(y_scaled, y_pred_scaled)
        print(f"[INFO] Valutazione surrogato → MSE: {mse:.4f} | R²: {r2:.4f}")
        
        # Ricalcola l'euristica
        self.compute_heuristic()

        # Dopo il retraining puoi anche stampare i nuovi valori previsti:
        X = np.array([lam[:2] for lam in self.Selected])
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        print("Nuove previsioni del surrogato sui Selected:")
        for lam, pred in zip(self.Selected, y_pred):
            print(f"  Lambda: {lam}, Predizione surrogato: {pred:.4f}")

def get_or_train_model(archive_file, model_path, n_training=100, random_state=42):
    """
    Carica un modello surrogato se esiste, altrimenti lo allena e lo salva.
    """
    
    if os.path.exists(model_path):
        print(f"Caricamento del modello salvato...")
        surrogate_model = joblib.load(model_path)
    else:
        print(f"Alleno un nuovo modello surrogato...")
        data = load_lambda_covariance_data(archive_file)
        surrogate_model, *_ = train_and_prepare_surrogate(data, n_training, random_state)
        joblib.dump(surrogate_model, model_path)
    return surrogate_model

def run_single_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None):
    print(f"Avvio colonia {colony_id}")
    aco = ACOActiveLearner(archive_data, gp_model)
    # Diversifica i lambda iniziali per ogni colonia (es: random seed diverso)
    print(f"Colonia {colony_id}: calcolo euristica")
    aco.compute_heuristic()
    print(f"Colonia {colony_id}: selezione iniziale diversificata")
    if already_selected is None:
        already_selected = set()
    aco.select_initial_diverse_lambdas(
        n_init_diverse=params['n_init_diverse'],
        random_state=colony_id,
        exclude_lambdas=already_selected
    )
    print(f"Colonia {colony_id}: avvio ACO")
    Selected, C_e, final_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=params['n_ants'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        beta=params['beta'],
        omega=params['omega'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        retrain_every=params['retrain_every'],
        n_init_diverse=params['n_init_diverse'],
        exclude_lambdas=already_selected,
        random_state=colony_id
    )
    return list(Selected)

def append_new_selected(lam, results_dict, shared_file="shared_selected.csv"):
    # results_dict: dict con info su lambda, sensitività, ecc.
    df = pd.DataFrame([results_dict])
    lock = FileLock(shared_file + ".lock")
    with lock:
        if not os.path.exists(shared_file):
            df.to_csv(shared_file, index=False)
        else:
            df.to_csv(shared_file, mode='a', header=False, index=False)

# per riaddestrare il modello surrogato globale
def retrain_loop(shared_file="shared_selected.csv", model_path="surrogate_global.pkl", interval=600):
    while True:
        lock = FileLock(shared_file + ".lock")
        with lock:
            if os.path.exists(shared_file):
                df = pd.read_csv(shared_file)
                model, scaler_X, scaler_y = fit_gp_model(df)
                joblib.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, model_path)
                print("Surrogato globale aggiornato.")
        time.sleep(interval)  # Attendi N secondi

def maybe_reload_surrogate(local_model, model_path="surrogate_global.pkl", last_mtime=None):
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if last_mtime is None or mtime > last_mtime:
            surrogate = joblib.load(model_path)
            print("Surrogato globale ricaricato.")
            return surrogate, mtime
    return local_model, last_mtime

# main.py

def main():
    print("Inizio script alternativa 3", flush=True)
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'
    
    archive_data=pd.read_csv(archive_file)
    print(f"Caricati {len(archive_data)} campioni da '{archive_file}'", flush=True)

    gp_model = get_or_train_model(archive_file, model_path, n_training=500)

    # Carica i dati di ground truth
    results_df = pd.read_csv(ground_truth_file, header=None)
    C_ref = results_df.values
    print(f"Shape di C_ref: {C_ref.shape}", flush=True)

    params = dict(
        n_ants=30,
        top_k=5,
        alpha=1.0,
        beta=1.0,
        omega=0.7,
        epsilon=1e-2,
        budget=30,
        retrain_every=5,
        n_init_diverse=7
    )

    n_colonies = 2  # Numero di colonie parallele
    # Avvia il loop di riaddestramento in un thread separato
    all_selected = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        futures = [
            executor.submit(run_single_colony, i, archive_data, gp_model, C_ref, params)
            for i in range(n_colonies)
        ]
        for future in concurrent.futures.as_completed(futures):
            all_selected.extend(future.result())

    # Rimuovi duplicati
    all_selected_unique = list({tuple(lam): lam for lam in all_selected}.values())
    print(f"Totale lambda selezionati (unici): {len(all_selected_unique)}")
    # Calcola la covarianza empirica e l'errore finale
    C_e_final = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(all_selected_unique)
    final_error = np.linalg.norm(C_ref - C_e_final, ord='fro') if C_e_final is not None else None
    print(f"Errore finale combinato: {final_error}")

    aco = ACOActiveLearner(archive_data, gp_model)
    aco.compute_heuristic()
    aco.select_initial_diverse_lambdas(n_init_diverse=params['n_init_diverse'])
    Selected, C_e, final_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=params['n_ants'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        beta=params['beta'],
        omega=params['omega'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        retrain_every=params['retrain_every'],
        n_init_diverse=params['n_init_diverse']
    )

    print(f"Totale lambda selezionati (unici): {len(Selected)}")
    print(f"Errore finale combinato: {final_error}")

    
    # Stampa la lista dei lambda ottimi identificati
    print("Lambda ottimi identificati (unione di tutte le colonie):")
    for lam in Selected:
        print(lam)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Errore durante l'esecuzione:", e, flush=True)
        traceback.print_exc()