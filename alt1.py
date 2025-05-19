# moaco_covariance.py
import numpy as np
import pandas as pd
from collections import defaultdict
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
import random
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations

def round_lambda(lam, ndigits=8):
    arr = np.asarray(lam)
    return tuple(np.round(arr.astype(float), ndigits))

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
            key = round_lambda(lam)
            self.eta[key] = y_pred[i]
            


    def select_initial_diverse_lambdas(self, n_init_diverse=20, random_state=None, exclude_lambdas=None):
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
            rng.shuffle(sorted_lambdas)
        else:
            rng = np.random.RandomState(GLOBAL_SEED)
            rng.shuffle(sorted_lambdas)

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

        for lam in selected:
            self.Selected.add(tuple(lam))
        print(f"Lambda iniziali selezionati (norma massima): {selected}")

    # campionamento dei valori di lamnda da assegnare alle formiche in base a tau e eta
    def sample_candidates(self, n_ants=20, alpha=1.0, beta=1.0):
        if not self.eta:
            raise ValueError("Euristica (eta) vuota: chiama prima compute_heuristic().")
        keys = list(self.eta.keys())
        scores = np.array([(self.tau[lam] ** alpha) * (self.eta[lam] ** beta) for lam in keys])
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-8
        probs = scores / np.sum(scores) if np.sum(scores) != 0 else np.ones_like(scores) / len(scores)
        n_sample = min(n_ants, len(keys))
        indices = np.random.choice(len(keys), size=n_sample, replace=False, p=probs)
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
            x_opt,_ = optimize_for_lambda(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar

        # Calcola C_e solo se ci sono abbastanza elementi
        if len(self.Selected) > 1:
            self.C_e = (deviations.T @ deviations) / (len(self.Selected) - 1)
        else:
            self.C_e = None

    def compute_Ce_from_lambdas(self,lambdas_list):
        if len(lambdas_list) <= 1:
            print(f'Numero insufficiente di lambda ({len(lambdas_list)}) per calcolare C_e.')
        
            # Restituisci una matrice nulla della dimensione giusta
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
            
    def fitness_moaco(self, lam, norm, C_ref, Selected, w1, w2):
        S_lam = norm[round_lambda(lam)]
        delta_k = self.compute_delta_k(C_ref, lam)
        # delta_k misura la variazione dell'errore aggiungendo lam
        # Se vuoi massimizzare la fitness, puoi invertire il segno di delta_k se necessario
        return w1 * S_lam - w2 * delta_k

    def optimize_weights(self, C_ref, Selected, candidates, norm, w_grid, k_frac=0.1):
        """
        Trova la combinazione di pesi (w1, w2) che minimizza la variazione media di covarianza
        tra i migliori candidati. Usa un parametro k dinamico: k = max(1, int(k_frac * len(candidates)))
        """
        best_score = np.inf
        best_weights = None
        k = max(1, int(k_frac * len(candidates)))
        for w1 in w_grid['w1']:
            for w2 in w_grid['w2']:
                fitness_list = [(lam, self.fitness_moaco(lam, norm, C_ref, Selected, w1, w2))
                                for lam in candidates]
                fitness_list.sort(key=lambda x: x[1], reverse=True)
                top_k = [lam for lam, _ in fitness_list[:k]]
                delta_sum = 0
                for lam in top_k:
                    C_e_old = self.compute_Ce_from_lambdas(Selected)
                    C_e_new = self.compute_Ce_from_lambdas(list(Selected) + [lam])
                    if C_e_old is not None and C_e_new is not None:
                        delta_sum += np.linalg.norm(C_e_new - C_e_old, ord='fro')
                if delta_sum < best_score:
                    best_score = delta_sum
                    best_weights = (w1, w2)
        return best_weights
            

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

def run_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None):
    print(f"Avvio colonia {colony_id}")
    learner= ACOActiveLearner(archive_data, gp_model)
    learner.compute_heuristic()
    learner.select_initial_diverse_lambdas(n_init_diverse=150, random_state=GLOBAL_SEED)
    Selected = set(learner.Selected)
    val_count = len(Selected)
    print(f"Colonia {colony_id}: selezione iniziale diversificata")
    if already_selected is None:
        already_selected = set()
    learner.select_initial_diverse_lambdas(
        n_init_diverse=20,
        random_state=GLOBAL_SEED+ colony_id,  # oppure GLOBAL_SEED + colony_id se in parallelo
        exclude_lambdas=already_selected
    )
    print(f"Colonia {colony_id}: avvio ACO, Lambda iniziali: {list(learner.Selected)}")
    Selected, C_e, final_error = learner.run_aco_active_learning(
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

def fitness_moaco(self,lam, norm, C_ref, Selected, w1, w2):
    S_lam = norm[round_lambda(lam)]
    C_e_proxy = self.compute_Ce_from_lambdas(list(Selected) + [lam])
    proxy_diff = np.linalg.norm(C_ref - C_e_proxy, ord='fro')
    return w1 * S_lam - w2 * proxy_diff

def optimize_weights(self,C_ref, Selected, candidates, norm, w_grid):
    best_score = np.inf
    best_weights = None
    for w1 in w_grid['w1']:
        for w2 in w_grid['w2']:
            fitness_list =  [(lam, fitness_moaco(self,lam, norm, C_ref, Selected, w1, w2))
                for lam in candidates]
            fitness_list.sort(key=lambda x: x[1], reverse=True)
            top_k = [lam for lam, _ in fitness_list[:5]]
            delta_sum = 0
            for lam in top_k:
                C_e_old = self.compute_Ce_from_lambdas(Selected)
                C_e_new = self.compute_Ce_from_lambdas(list(Selected) + [lam])
                delta_k = np.linalg.norm(C_ref - C_e_new, ord='fro') - np.linalg.norm(C_ref - C_e_old, ord='fro')
                delta_sum += abs(delta_k)
            if delta_sum < best_score:
                best_score = delta_sum
                best_weights = (w1, w2)
    return best_weights


def main():
    print("Inizio script MOACO alterativa 1")
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    C_ref = pd.read_csv(ground_truth_file, header=None).values

    surrogate_model = get_or_train_model(archive_file, model_path, n_training=500)
    learner = ACOActiveLearner(archive_data, surrogate_model)
    learner.compute_heuristic()
    # Popola norm con le stesse chiavi di self.eta
    norm = learner.eta.copy()
    # 1. Inizializzazione
    learner.select_initial_diverse_lambdas(n_init_diverse=150, random_state=GLOBAL_SEED)
    Selected = set(learner.Selected)
    val_count = len(Selected)
    Lambda = [tuple(row) for row in archive_data[['lambda1', 'lambda2', 'lambda3']].values]
    C_e = np.zeros_like(C_ref)
    tau = defaultdict(lambda: 1.0)
    w1, w2 = 1.0, 1.0

    n_ants = 150
    top_k = 50
    retrain_every = 5
    budget = 30
    alpha = 1.0
    beta = 1.0

    error_list = []
    w_grid = {'w1': np.linspace(0.5, 2.0, 4), 'w2': np.linspace(0.5, 2.0, 4)}

    iter_num = 0

    while val_count > budget:
        print(f"\nIterazione {val_count+1}/{budget}")

        # 1. Evaporazione del feromone
        evaporation_rate = 0.8
        for lam in Selected:
            tau[lam] *= evaporation_rate

        # Definisci i candidati prima di ottimizzare i pesi
        candidates = list(Selected)  # oppure usa sample_candidates se preferisci

        # 2. Calcola pesi ottimali per la fitness
        w1, w2 = learner.optimize_weights(C_ref, Selected, candidates, norm, w_grid, k_frac=0.1)
        print(f"Pesi ottimali trovati: w1={w1}, w2={w2}")

        # 3. Calcola fitness e aggiorna feromone SOLO sui lambda selezionati
        fitness_dict = {}
        for lam in Selected:
            fit = learner.fitness_moaco(lam, norm, C_ref, Selected, w1, w2)
            fitness_dict[lam] = fit
            tau[lam] += fit  # deposito di feromone proporzionale alla fitness

        # Aggiorna C_e
        C_e = learner.compute_Ce_from_lambdas(Selected)
        if C_e is not None:
            error = np.linalg.norm(C_ref - C_e, ord='fro')
        else:
            error = None
        error_list.append(error)
        print(f"Errore attuale dopo batch: {error:.4f}")

        # 3. Pruning: elimina lambda con fitness/feromone più basso e troppo vicini
        if len(Selected) > budget:
            print(f"Pruning: mantengo solo i {budget} lambda migliori e distanti almeno min_dist.")
            min_dist = 0.05  # scegli il valore più adatto al tuo problema
            sorted_lambdas = sorted(list(Selected), key=lambda l: tau[l], reverse=True)
            pruned = []
            for lam in sorted_lambdas:
                lam_arr = np.array(lam)
                if all(np.linalg.norm(lam_arr - np.array(other)) >= min_dist for other in pruned):
                    pruned.append(lam)
                if len(pruned) >= budget:
                    break
            Selected = set(pruned)
            # Rimuovi anche dal dizionario tau quelli eliminati
            for lam in list(tau.keys()):
                if lam not in Selected:
                    del tau[lam]

        # Aggiorna surrogato ogni retrain_every iterazioni
        iter_num += 1
        if iter_num % retrain_every == 0:
            pass

        val_count = len(Selected)
        if error is not None and error < 1e-3:
            print("Errore sufficientemente piccolo, termino.")
            break

    print("\n==== RISULTATI FINALI ====")
    print(f"Selected points: {len(Selected)}")
    print(f"Final error: {error:.6f}")
    print(f"Pesi ottimali finali: w1={w1}, w2={w2}")
    for lam in Selected:
        print(f"  {lam}")
    print("lambda selezionati (unici):",Selected)
    # Stampa le due matrici in modo leggibile
    print("\nMatrice C_ref (Ground Truth):")
    print(np.array2string(C_ref, precision=4, suppress_small=True))

    print("\nMatrice C_e finale (Empirica):")
    if C_e is not None:
        print(np.array2string(C_e, precision=4, suppress_small=True))
    else:
        print("C_e non disponibile.")

    plt.plot(error_list)
    plt.xlabel("Iterazione")
    plt.ylabel("Errore Frobenius")
    plt.title("Andamento dell'errore per iterazione")
    plt.grid(True)
    plt.savefig("errore_moaco.png")
    print("Plot salvato come 'errore_moaco.png'")

if __name__ == "__main__":
    main()