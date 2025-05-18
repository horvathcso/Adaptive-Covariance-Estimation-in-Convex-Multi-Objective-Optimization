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
        else:
            self.C_e = None

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

def run_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None):
    print(f"Avvio colonia {colony_id}")
    learner= ACOActiveLearner(archive_data, gp_model)
    # Diversifica i lambda iniziali per ogni colonia (es: random seed diverso)
    print(f"Colonia {colony_id}: calcolo euristica")
    learner.compute_heuristic()
    print(f"Colonia {colony_id}: selezione iniziale diversificata")
    if already_selected is None:
        already_selected = set()
    learner.select_initial_diverse_lambdas(
        n_init_diverse=params['n_init_diverse'],
        random_state=colony_id  + GLOBAL_SEED,
        exclude_lambdas=already_selected
    )
    print(f"Colonia {colony_id}: avvio ACO, Lambda iniziali: {list(learner.Selected)}")
    error_list = []
    for it in range(params['max_iter']):
        top_candidates = run_aco_phase(learner, params['n_ants'], params['alpha'], params['beta'], params['top_k'])
        informative_candidates = run_active_learning_phase(C_ref, learner, top_candidates)
        for lam in informative_candidates:
            if lam not in learner.Selected:
                _, x_opt = optimize_for_lambda(lam)
                x_opt_flat = np.asarray(x_opt).flatten()
                sens_norm = np.linalg.norm(x_opt_flat)
                learner.Selected.add(lam)
                learner.tau[lam] += sens_norm
                learner.eta[lam] = sens_norm
        learner.update_selected_and_Ce(lam)
        C_e = learner.C_e
        if C_e is not None:
            error = np.linalg.norm(C_ref - C_e, ord='fro')
            error_list.append(error)
            if error < 1e-3:
                break
        learner.compute_heuristic()
    return list(learner.Selected), learner.C_e

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

def run_aco_phase(learner, n_ants, alpha, beta, top_k):
    # Campiona candidati secondo ACO
    candidates = learner.sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)
    # Ordina per norma surrogata
    sorted_candidates = sorted(candidates, key=lambda lam: learner.eta[lam], reverse=True)
    return sorted_candidates[:top_k]

def run_active_learning_phase(C_ref, learner, candidates):
    delta_k_list = []
    for lam in candidates:
        delta_k = learner.compute_delta_k(C_ref, lam)
        delta_k_list.append((lam, delta_k))
    delta_k_list.sort(key=lambda x: x[1])
    return [lam for lam, _ in delta_k_list]

def main():
    print("Inizio script alternativa 2")
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    C_ref = pd.read_csv(ground_truth_file, header=None).values

    surrogate_model = get_or_train_model(archive_file, model_path, n_training=500)
    
    n_colonies = 4
    params = {
        'n_init_diverse': 7,
        'max_iter': 30,
        'n_ants': 30,
        'top_k': 5,
        'alpha': 1.0,
        'beta': 1.0,
    }
    
    # Per evitare duplicati iniziali, tieni traccia dei lambda già scelti
    already_selected = set()
    futures = []
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        for colony_id in range(n_colonies):
            # Passa una copia degli already_selected per ogni colonia
            futures.append(
                executor.submit(
                    run_colony,
                    colony_id,
                    archive_data,
                    surrogate_model,
                    C_ref,
                    params,
                    already_selected.copy()
                )
            )
            # Aggiorna already_selected con i lambda scelti da questa colonia (solo per la prossima)
            # NOTA: questa logica funziona bene solo se le colonie partono in sequenza, non in parallelo.
            # Per garantire unicità, puoi prima selezionare i lambda iniziali in sequenza, poi lanciare le colonie in parallelo.
            # Qui sotto una versione robusta:
        # Prima scegli i lambda iniziali in sequenza
        initial_lambdas_per_colony = []
        learner_tmp = ACOActiveLearner(archive_data, surrogate_model)
        learner_tmp.compute_heuristic()
        for colony_id in range(n_colonies):
            learner_tmp.select_initial_diverse_lambdas(
                n_init_diverse=params['n_init_diverse'],
                random_state=colony_id + GLOBAL_SEED,
                exclude_lambdas=already_selected
            )
            initial_lambdas = list(learner_tmp.Selected - already_selected)
            initial_lambdas_per_colony.append(initial_lambdas)
            already_selected.update(initial_lambdas)
        # Ora lancia le colonie in parallelo, ognuna con i suoi lambda iniziali
        futures = []
        for colony_id in range(n_colonies):
            futures.append(
                executor.submit(
                    run_colony,
                    colony_id,
                    archive_data,
                    surrogate_model,
                    C_ref,
                    params,
                    set(initial_lambdas_per_colony[colony_id])
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Unisci i lambda selezionati da tutte le colonie ed elimina duplicati
    all_selected = []
    all_Ce = []
    for selected, C_e in results:
        all_selected.extend(selected)
        all_Ce.append(C_e)
    # Elimina duplicati
    all_selected_unique = [tuple(lam) for lam in {tuple(np.round(lam, 8)) for lam in all_selected}]
    print(f"\n[INFO] Lambda selezionati totali (unici): {len(all_selected_unique)}")
    print(all_selected_unique)

    # Calcola la matrice empirica finale C_e_union
    def compute_Ce_from_lambdas(lambdas_list):
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

    C_e_union = compute_Ce_from_lambdas(all_selected_unique)
    error_union = np.linalg.norm(C_ref - C_e_union, ord='fro') if C_e_union is not None else None

    print("\n--- RISULTATI FINALI MULTICOLONY ---")
    print(f"Numero lambda selezionati (unici): {len(all_selected_unique)}")
    print(f"Errore finale (unione): {error_union:.4f}")
    print(f"\nMatrice C_ref:\n{C_ref}")
    print(f"\nMatrice C_e_union:\n{C_e_union}")
    
    
    ''' Caso NO colonie 
    learner = ACOActiveLearner(archive_data, surrogate_model)
    learner.compute_heuristic()

    # Inizializzazione lambda iniziali (alta norma)
    learner.select_initial_diverse_lambdas(n_init_diverse=7, random_state=GLOBAL_SEED)
    print(f"Lambda iniziali selezionati: {list(learner.Selected)}")

    max_iter = 30
    n_ants = 30
    top_k = 5
    alpha = 1.0
    beta = 1.0

    error_list = []
    for it in range(max_iter):
        print(f"\n[INFO] Iterazione {it+1}/{max_iter}")

        # --- Fase 1: ACO (esplorazione con surrogato) ---
        top_candidates = run_aco_phase(learner, n_ants, alpha, beta, top_k)
        print(f"Top-{top_k} candidati ACO: {top_candidates}")

        # --- Fase 2: Active Learning (minimizzazione errore) ---
        informative_candidates = run_active_learning_phase(C_ref, learner, top_candidates)
        print(f"Candidati ordinati per riduzione errore: {informative_candidates}")

        # --- Valutazione esatta e aggiornamento ---
        for lam in informative_candidates:
            if lam not in learner.Selected:
                _, x_opt = optimize_for_lambda(lam)
                x_opt_flat = np.asarray(x_opt).flatten()
                sens_norm = np.linalg.norm(x_opt_flat)
                learner.Selected.add(lam)
                learner.tau[lam] += sens_norm
                learner.eta[lam] = sens_norm
                print(f"[DEBUG] valutato: lambda={lam}, norm={sens_norm:.4f}")

        learner.update_selected_and_Ce(lam)
        C_e = learner.C_e
        if C_e is not None:
            error = np.linalg.norm(C_ref - C_e, ord='fro')
            error_list.append(error)
            print(f"Errore attuale dopo batch: {error:.4f}")
            if error < 1e-3:
                print("Errore sufficientemente piccolo, termino.")
                break
        else:
            print("Errore attuale: N/A (C_e non definita)")

        learner.compute_heuristic()

    print("\n--- RISULTATI FINALI ---")
    print(f"Selected: {learner.Selected}")
    print(f"Errore finale: {error_list[-1] if error_list else 'N/A'}")
    print(f"C_e finale:\n{C_e}")'''

if __name__ == "__main__":
    main()