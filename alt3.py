# aco_active_learning.py
import pandas as pd
import numpy as np
import random
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

GLOBAL_SEED = 42 # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations

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
     

    def select_initial_diverse_lambdas(self, n_init_diverse=5, random_state=None, exclude_lambdas=None):
        """
        Seleziona n_init_diverse lambda iniziali massimizzando la diversità (distanza euclidea).
        Garantisce che ogni lambda sia nell'intervallo [0, 1] e la somma sia 1.
        """
        def normalize_lambda(lam):
            lam = np.clip(lam, 0, 1)
            return lam / np.sum(lam)

        if exclude_lambdas is None:
            exclude_lambdas = set()
        if not self.eta:
            raise ValueError("Euristica (eta) non calcolata. Chiama compute_heuristic() prima.")

        sorted_lambdas = sorted(self.eta.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState(GLOBAL_SEED)
        rng.shuffle(sorted_lambdas)
        print(f"Lambda ordinati per euristica (norma): {len(sorted_lambdas)}")
        selected = []
        seen = set()
        for lam, score in sorted_lambdas:
            lam_arr = np.array(lam)
            lam_arr = normalize_lambda(lam_arr)
            lam_tuple = tuple(lam_arr)
            if lam_tuple not in seen and lam_tuple not in exclude_lambdas:
                selected.append(lam_arr)
                seen.add(lam_tuple)
            if len(selected) >= n_init_diverse:
                break
            # Se non bastano, genera lambda casuali
        n_dim = 3
        while len(selected) < n_init_diverse:
            lam_arr = rng.dirichlet(np.ones(n_dim))
            lam_tuple = tuple(np.round(lam_arr, 8))
            if lam_tuple not in seen and lam_tuple not in exclude_lambdas:
                selected.append(lam_arr)
                seen.add(lam_tuple)
            print(f"Lambda casuali generati: {len(lam_tuple)}")

        for lam in selected:
            self.Selected.add(tuple(lam))
        print(f"Lambda iniziali selezionati (norma massima): {selected}")

    # campionamento dei valori di lamnda da assegnare alle formiche in base a tau e eta
    def sample_candidates(self, n_ants=100, alpha=1.0, beta=1.0, random_state=None):
           
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
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState(GLOBAL_SEED)
        indices = rng.choice(len(keys), size=n_ants, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        print(f"Candidati campionati: {sampled}")
        return sampled

    # Aggiorna la lista di lambda selezionati e calcola la matrice di covarianza empirica C_e
    def update_selected_and_Ce(self, selected_lambda=None, C_lambda=None):
        if selected_lambda is not None:
            self.Selected.add(tuple(selected_lambda))
        #print("Lambda attualmente in Selected:")
        #for lam in self.Selected:
            #print(lam)
        if len(self.Selected) <= 1:
            print("Numero insufficiente di elementi selezionati per calcolare C_e.")
            self.C_e = None
            return

        x_list = []
        for lam in self.Selected:
            x_opt, _ = optimize_for_lambda(lam)
            x_opt = np.asarray(x_opt).flatten()
            if x_opt.shape[0] != 3:
                print(f"[ERRORE] x_opt per lambda {lam} ha shape {x_opt.shape} invece di (3,)")
            x_list.append(x_opt)  # x_opt deve essere shape (3,)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar

        # Calcola C_e solo se ci sono abbastanza elementi
        if len(self.Selected) > 1:
            self.C_e = (deviations.T @ deviations) / (len(self.Selected) - 1)
            print(f"Nuova matrice C_e calcolata:\n{self.C_e}")
            print(f"Shape di C_e: {self.C_e.shape}")
        else:
            self.C_e = None
       

    def run_aco_active_learning(self, C_ref, archive_data, n_ants=150,
                                alpha=1.0, beta=1.0, omega=0.7,
                                epsilon=None, budget=20, retrain_every=6,
                                n_init_diverse=5, exclude_lambdas=None, random_state=None,
                                min_lambda=10):
        if epsilon is None:
            epsilon = 1e-6  # errore estremamente piccolo
        print (f"Avvio ACO Active Learning con budget {budget} e epsilon {epsilon}", flush=True)
        # Selezione iniziale diversificata SOLO se Selected è vuoto
        if len(self.Selected) == 0:
            self.select_initial_diverse_lambdas(
                n_init_diverse=n_init_diverse,
                random_state=random_state,
                exclude_lambdas=exclude_lambdas
            )
            self.update_selected_and_Ce(next(iter(self.Selected)))
        
        val_count = len(self.Selected)
        print(f"Numero iniziale di lambda selezionati: {val_count}", flush=True)
        print(f"[DEBUG] Prima del ciclo: val_count={val_count}, len(self.Selected)={len(self.Selected)}, badget={budget}", flush=True)
        error_list = []
        max_iter = budget  # oppure usa un parametro dedicato, es: max_iter=params['max_iter']
        for iter_idx in range(max_iter):
            print(f"Iterazione {iter_idx + 1}/{max_iter}", flush=True)
            print(f"[DEBUG] Iterazione {iter_idx + 1}: numero di lambda selezionati nella colonia = {len(self.Selected)}")
            # 1. Valuta solo i lambda già selezionati
            delta_k_list = []
            for lam in self.Selected:
                print(f"Calcolo delta_k per lambda {lam}", flush=True)
                delta_k = self.compute_delta_k(C_ref, lam)
                print(f"Delta_k per lambda {lam}: {delta_k:.8f}", flush=True)
                delta_k_list.append((lam, delta_k))
            # 2. Ordina per miglioramento (delta_k più negativo = maggiore riduzione errore)
            delta_k_list.sort(key=lambda x: x[1])
            selected_candidates = [lam for lam, _ in delta_k_list]
            print(f"Candidati selezionati per update: {len(selected_candidates)}", flush=True)
            # 3. Evaporazione del feromone su tutti i lambda
            self.evaporate_pheromones()
            # 4. Deposito feromone su tutti i candidati valutati (proporzionale alla riduzione errore)
            self.update_pheromones(selected_candidates, C_ref)
            # 5. Aggiorna Selected: aggiungi tutti i candidati valutati
            for lam in selected_candidates:
                self.Selected.add(tuple(lam))
            print(f"Lambda selezionati dopo l'iterazione: {len(self.Selected)}", flush=True)
            # 6. Pruning: tieni solo le formiche con più feromone e sufficientemente diverse
            self.prune_selected_lambdas(min_lambda=20, min_dist=0.05, random_state=random_state)
            print(f"Lambda selezionati dopo il pruning: {len(self.Selected)}", flush=True)
            # 7. Aggiorna C_e e calcola errore
            self.update_selected_and_Ce(None)
            if self.C_e is not None:
                error = np.linalg.norm(C_ref - self.C_e, ord='fro')
            else:
                error = None
            error_list.append(error)
            print(f"Errore attuale dopo iterazione: {error:.8f}" if error is not None else "Errore attuale: N/A")
            # Condizione di arresto: errore molto piccolo o pochi campioni rimasti
            if (error is not None and error < epsilon) or (len(self.Selected) < min_lambda):
                print(f"STOP: errore < {epsilon} o numero campioni < {min_lambda}")
                break

        plt.figure(figsize=(8,4))
        plt.plot(error_list, marker='o')
        plt.title("Errore Frobenius tra $C_{ref}$ e $C_e$ per iterazione")
        plt.xlabel("Iterazione")
        plt.ylabel("Errore Frobenius")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("errore_per_iterazione.png")
        plt.show()
        return list(self.Selected), self.C_e, error_list[-1] if error_list else None


    def compute_Ce_from_lambdas(self, lambdas_list):
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            x_opt,_ = optimize_for_lambda(lam)
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
    
    
    def evaporate_pheromones(self):
        for lam in self.tau:
            self.tau[lam] *= (1 - self.rho)

    def update_pheromones(self, selected_lambdas, C_ref):
        # Calcola la riduzione di errore per ogni lambda selezionato
        for lam in selected_lambdas:
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected - {lam}))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected))
            if C_e_old is not None and C_e_new is not None:
                error_old = np.linalg.norm(C_ref - C_e_old, ord='fro')
                error_new = np.linalg.norm(C_ref - C_e_new, ord='fro')
                delta_error = error_old - error_new
                # Deposita feromone proporzionale alla riduzione di errore (solo se positivo)
                if delta_error > 0:
                    self.tau[lam] += delta_error

    def prune_selected_lambdas(self, min_lambda=20, min_dist=0.05,random_state=None):
        """
        Pruning dei lambda selezionati: tiene i min_lambda migliori per feromone,
        imponendo che ogni lambda sia almeno a distanza min_dist dagli altri.
        """
        # Ordina per feromone decrescente
        sorted_lambdas = sorted(self.Selected, key=lambda lam: self.tau[lam], reverse=True)
        pruned = []
        for lam in sorted_lambdas:
            lam_arr = np.array(lam)
            if all(np.linalg.norm(lam_arr - np.array(other)) >= min_dist for other in pruned):
                pruned.append(lam)
            if len(pruned) >= min_lambda:
                break
        self.Selected = set(pruned)
        self.C_e = self.compute_Ce_from_lambdas(pruned)
        
    # NON LO STO USANDO
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

def run_single_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None, random_state=None):
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
        random_state=random_state,
        exclude_lambdas=already_selected
    )
    print(f"Colonia {colony_id}: avvio ACO, lambda iniziali :{list(aco.Selected)}")
    Selected, C_e, final_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=params['n_ants'],
        alpha=params['alpha'],
        beta=params['beta'],
        omega=params['omega'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        retrain_every=params['retrain_every'],
        n_init_diverse=params['n_init_diverse'],
        exclude_lambdas=already_selected,
        random_state=random_state
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
        n_ants=150,
        top_k=50,
        alpha=1.0,
        beta=1.0,
        omega=0.7,
        epsilon=1e-2,
        budget=30,
        retrain_every=5,
        n_init_diverse=50  # (numero di capioni di lambda iniziali)
    )

    n_colonies = 2  # Numero di colonie parallele
    # Avvia il loop di riaddestramento in un thread separato
    all_selected = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        futures = [
            executor.submit(
                run_single_colony,
                i,
                archive_data,
                gp_model,
                C_ref,
                params,
                random_state=GLOBAL_SEED + i  # Passa il seed deterministico
            )
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
    print ("Lambda selezionati (unione di tutte le colonie):" )
    print(all_selected_unique)
    # Stampa le due matrici in modo leggibile
    print("\nMatrice C_ref (Ground Truth):")
    print(np.array2string(C_ref, precision=4, suppress_small=True))

    print("\nMatrice C_e_final (Empirica):")
    if C_e_final is not None:
        print(np.array2string(C_e_final, precision=4, suppress_small=True))
    else:
        print("C_e_final non disponibile.")

    '''aco = ACOActiveLearner(archive_data, gp_model)
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
        print(lam)'''
    # 1. Plot degli errori finali di ogni colonia
    colony_errors = []
    for i in range(n_colonies):
        # Ricalcola l'errore per ogni colonia
        colony_selected = futures[i].result()
        C_e_colony = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(colony_selected)
        err = np.linalg.norm(C_ref - C_e_colony, ord='fro') if C_e_colony is not None else None
        colony_errors.append(err)

    plt.figure(figsize=(6,4))
    plt.bar(range(n_colonies), colony_errors)
    plt.xlabel("Colonia")
    plt.ylabel("Errore Frobenius finale")
    plt.title("Errore finale per colonia")
    plt.tight_layout()
    plt.savefig("final_error_per_colony.png")
    plt.show()

    # 2. Visualizzazione C_ref e C_e_final
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(C_ref, aspect='auto', cmap='viridis')
    plt.title("C_ref (Ground Truth)")
    plt.colorbar()
    plt.subplot(1,2,2)
    if C_e_final is not None and len(C_e_final.shape) == 2:
        plt.imshow(C_e_final, aspect='auto', cmap='viridis')
        plt.title("C_e_final (Empirica)")
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, "C_e_final non disponibile", ha='center', va='center')
        plt.title("C_e_final (Empirica)")
    plt.tight_layout()
    plt.savefig("Cref_vs_Cefinal.png")
    plt.show()
    print("\n--- RISULTATI FINALI MULTICOLONY ---")
    print(f"Numero lambda selezionati (unici): {len(all_selected_unique)}")
    print(f"Errore finale (unione): {err:.4f}")
    print(f"\nMatrice C_ref:\n{C_ref}")
    print(f"\nMatrice C_e_union:\n{C_e_final}")
    print("Norma Frobenius di C_e_union:", np.linalg.norm(C_ref - C_e_final, ord='fro'))
    print("Shape di C_e_union:", C_e_final.shape)
    


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Errore durante l'esecuzione:", e, flush=True)
        traceback.print_exc()