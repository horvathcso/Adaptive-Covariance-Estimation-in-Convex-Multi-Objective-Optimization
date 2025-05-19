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

NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations
GLOBAL_SEED = 42 # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None, rho=0.1):
        self.lambda_data = lambda_data.copy()
        self.lambda_data['lambda3'] = 1 - self.lambda_data['lambda1'] - self.lambda_data['lambda2']
        self.lambdas = self.lambda_data[['lambda1', 'lambda2', 'lambda3']].values

        self.tau = defaultdict(lambda: 1.0)
        self.eta_surrogate = {}  # euristica dal surrogato
        self.eta_true = {}       # euristica esatta (norma sensibilità vera)
        self.surrogate = surrogate_model

        self.Selected = set()
        self.C_e = None

        self.epsilon_threshold = 1e-2
        self.rho = rho

    def compute_heuristic(self):
        if self.surrogate is None:
            raise ValueError("Surrogate model non definito.")
        print("Calcolo euristica surrogata per ogni lambda...")
        X = self.lambda_data[['lambda1', 'lambda2']].values
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        for i, lam in enumerate(self.lambdas):
            self.eta_surrogate[tuple(lam)] = y_pred[i]

    def select_initial_diverse_lambdas(self, n_init_diverse, random_state=None, exclude_lambdas=None):
        """
        Seleziona n_init_diverse lambda iniziali massimizzando la diversità (distanza euclidea).
        Se non ci sono abbastanza lambda disponibili, genera i restanti casualmente.
        """
        def normalize_lambda(lam):
            lam = np.clip(lam, 0, 1)
            return lam / np.sum(lam)

        if exclude_lambdas is None:
            exclude_lambdas = set()
        if not self.eta_surrogate:
            raise ValueError("Euristica (eta) non calcolata. Chiama compute_heuristic() prima.")

        sorted_lambdas = sorted(self.eta_surrogate.items(), key=lambda x: x[1], reverse=True)
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

        # Se non bastano, genera lambda casuali
        n_dim = 3
        while len(selected) < n_init_diverse:
            lam_arr = rng.dirichlet(np.ones(n_dim))
            lam_tuple = tuple(np.round(lam_arr, 8))
            if lam_tuple not in seen and lam_tuple not in exclude_lambdas:
                selected.append(lam_arr)
                seen.add(lam_tuple)

        for lam in selected:
            self.Selected.add(tuple(lam))
        print(f"Lambda iniziali selezionati (norma massima o casuali): {len(selected)}")

    def sample_candidates(self, n_ants=100, alpha=1.0, beta=1.0):
        print("Campionamento candidati...")
        keys = list(self.eta_surrogate.keys())
        scores = np.array([
            (self.tau[lam] ** alpha) * (self.eta_surrogate[lam] ** beta) for lam in keys
        ])
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-8
        if np.sum(scores) == 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / np.sum(scores)
        n_sample = min(n_ants, len(keys))  # Fix: non campionare più di quanto disponibile
        indices = np.random.choice(len(keys), size=n_sample, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        return sampled

    def update_selected_and_Ce(self, selected_lambda):
        if selected_lambda is not None:
            self.Selected.add(tuple(selected_lambda))
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
        if len(self.Selected) > 1:
            self.C_e = (deviations.T @ deviations) / (len(self.Selected) - 1)
            print(f"Nuova matrice C_e calcolata:\n{self.C_e}")
        else:
            self.C_e = None

    def compute_Ce_from_lambdas(self, lambdas_list):
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            x_opt,_ = optimize_for_lambda(lam)
            x_opt = np.asarray(x_opt).flatten()
            if x_opt.shape[0] != 3:
                print(f"[ERRORE] x_opt per lambda {lam} ha shape {x_opt.shape} invece di (3,)")
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        C_e = (deviations.T @ deviations) / (len(lambdas_list) - 1)
        return C_e

    def compute_delta_k(self, C_ref, lam):
        if not self.Selected:
            C_e_single = self.compute_Ce_from_lambdas([lam])
            delta_k = -np.linalg.norm(C_ref - C_e_single, ord='fro')
        else:
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected) + [lam])
            delta_k = np.linalg.norm(C_ref - C_e_new, ord='fro') - np.linalg.norm(C_ref - C_e_old, ord='fro')
        return delta_k
    # NON LO USO
    def retrain_surrogate(self, archive_data):
        print("Riaddestro il modello surrogato con i seguenti nuovi lambda:")
        for lam in self.Selected:
            print(f"  {lam}")
        selected_records = []
        for lam in self.Selected:
            cov_matrix, x_opt = hessian_estimation_for_lambda(lam)
            Sigma_x, Sigma_f = estimate_local_covariances_from_lambdas(lambda_vec=lam, num_perturbations=NUM_PERTURBATIONS, delta=PERTURBATION_STRENGTH)
            try:
                P = np.linalg.cholesky(cov_matrix).T
                P_flattened = P[np.triu_indices_from(P)]
            except np.linalg.LinAlgError:
                P = np.full_like(cov_matrix, np.nan)
                P_flattened = np.full((cov_matrix.shape[0] * (cov_matrix.shape[0] + 1)) // 2, np.nan)
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
        selected_df = pd.DataFrame(selected_records)
        combined_df = pd.concat([archive_data, selected_df], ignore_index=True)
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
        self.compute_heuristic()
        X = np.array([lam[:2] for lam in self.Selected])
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        print("Nuove previsioni del surrogato sui Selected:")
        #for lam, pred in zip(self.Selected, y_pred):
            #print(f"  Lambda: {lam}, Predizione surrogato: {pred:.4f}")

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

    def prune_selected_lambdas(self, min_lambda=20, min_dist=0.05):
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

def get_or_train_model(archive_file, model_path, n_training=100, random_state=42):
    if os.path.exists(model_path):
        print(f"Caricamento del modello salvato...")
        surrogate_model = joblib.load(model_path)
    else:
        print(f"Alleno un nuovo modello surrogato...")
        data = load_lambda_covariance_data(archive_file)
        surrogate_model, *_ = train_and_prepare_surrogate(data, n_training, random_state)
        joblib.dump(surrogate_model, model_path)
    return surrogate_model


def run_aco_phase(learner, n_ants, alpha, beta):
    return learner.sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)

def run_active_learning_phase(C_ref, learner, candidates):
    delta_k_list = []
    for lam in candidates:
        delta_k = learner.compute_delta_k(C_ref, lam)
        delta_k_list.append((lam, delta_k))
    delta_k_list.sort(key=lambda x: x[1])
    return [lam for lam, _ in delta_k_list]


def run_colony(colony_id, archive_data, gp_model, C_ref, params, initial_lambdas):
    print(f"Avvio colonia {colony_id}")
    learner = ACOActiveLearner(archive_data, gp_model)
    learner.compute_heuristic()
    # Imposta i lambda iniziali già selezionati
    learner.Selected = set(initial_lambdas)
    print(f"Colonia {colony_id}: Lambda iniziali: {list(learner.Selected)}")
    error_list = []
    for it in range(params['max_iter']):
        print(f"\n[Colonia {colony_id}] Iterazione {it+1}/{params['max_iter']}")
        # --- Fase 1: ACO (esplorazione con surrogato) ---
        candidates = run_aco_phase(learner, params['n_ants'], params['alpha'], params['beta'])
        print(f"[Colonia {colony_id}] Candidati campionati (ACO): {candidates}")
        # Filtra solo i candidati non ancora selezionati
        candidates = [lam for lam in candidates if lam not in learner.Selected]
        print(f"[Colonia {colony_id}] Candidati NON ancora selezionati: {candidates}")
        if not candidates:
            print(f"[Colonia {colony_id}] Nessun nuovo candidato disponibile, interrompo la colonia.")
            break
        # --- Fase 2: Active Learning (minimizzazione errore) ---
        informative_candidates = run_active_learning_phase(C_ref, learner, candidates)
        print(f"[Colonia {colony_id}] Candidati ordinati per riduzione errore: {informative_candidates}")
        # Filtra solo i candidati non ancora selezionati
        informative_candidates = [lam for lam in informative_candidates if lam not in learner.Selected]
        print(f"[Colonia {colony_id}] Candidati informativi NON ancora selezionati: {informative_candidates}")
        for lam in informative_candidates:
            if lam not in learner.Selected:
                x_opt,_ = optimize_for_lambda(lam)
                x_opt_flat = np.asarray(x_opt).flatten()
                sens_norm = np.linalg.norm(x_opt_flat)
                learner.Selected.add(lam)
                print(f"[Colonia {colony_id}] [DEBUG] Lambda selezionato: {lam}, norm={sens_norm:.4f}, totale selezionati: {len(learner.Selected)}")
                learner.tau[lam] += sens_norm
                learner.eta_true[lam] = sens_norm  # salva la norma vera

        # --- PRUNING: mantieni solo i lambda migliori ---
        min_lambda = 20  # numero fisso di lambda da mantenere
        if len(learner.Selected) > min_lambda:
            print(f"[Colonia {colony_id}] Pruning: seleziono i {min_lambda} lambda migliori e diversi.")
            learner.prune_selected_lambdas(min_lambda=min_lambda, min_dist=0.05)
            print(f"[Colonia {colony_id}] Dopo pruning, Selected: {learner.Selected}")

        # Aggiorna C_e e calcola errore
        if informative_candidates:
            learner.update_selected_and_Ce(informative_candidates[-1])
        else:
            learner.update_selected_and_Ce(None)
        C_e = learner.C_e
        if C_e is not None:
            error = np.linalg.norm(C_ref - C_e, ord='fro')
            error_list.append(error)
            print(f"[Colonia {colony_id}] Iterazione {it+1}: Errore attuale = {error:.4f}")
            # --- CONDIZIONE DI STOP ---
            if error < 1e-3 or len(learner.Selected) <= min_lambda:
                print(f"[Colonia {colony_id}] STOP: errore < 1e-3 o numero lambda <= {min_lambda}")
                break
        else:
            print(f"[Colonia {colony_id}] Errore attuale: N/A (C_e non definita)")

        learner.compute_heuristic()
        print(f"[Colonia {colony_id}] Euristica aggiornata.")
        # Evapora feromone
        learner.evaporate_pheromones()
        print(f"[Colonia {colony_id}] Feromone evaporato.")
        # Aggiorna feromone solo sui candidati valutati
        learner.update_pheromones(informative_candidates, C_ref)
        print(f"[Colonia {colony_id}] Feromone aggiornato.")

    print(f"[Colonia {colony_id}] Fine colonia. Selected finali: {learner.Selected}")
    return list(learner.Selected), learner.C_e, error_list

def append_new_selected(lam, results_dict, shared_file="shared_selected.csv"):
    df = pd.DataFrame([results_dict])
    lock = FileLock(shared_file + ".lock")
    with lock:
        if not os.path.exists(shared_file):
            df.to_csv(shared_file, index=False)
        else:
            df.to_csv(shared_file, mode='a', header=False, index=False)

def retrain_loop(shared_file="shared_selected.csv", model_path="surrogate_global.pkl", interval=600):
    while True:
        lock = FileLock(shared_file + ".lock")
        with lock:
            if os.path.exists(shared_file):
                df = pd.read_csv(shared_file)
                model, scaler_X, scaler_y = fit_gp_model(df)
                joblib.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, model_path)
                print("Surrogato globale aggiornato.")
        time.sleep(interval)

def maybe_reload_surrogate(local_model, model_path="surrogate_global.pkl", last_mtime=None):
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if last_mtime is None or mtime > last_mtime:
            surrogate = joblib.load(model_path)
            print("Surrogato globale ricaricato.")
            return surrogate, mtime
    return local_model, last_mtime


def main():
    print("Inizio script alternativa 2")
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    C_ref = pd.read_csv(ground_truth_file, header=None).values
    print(f"Caricamento matrice C_ref da {ground_truth_file}:\n{C_ref}")
    surrogate_model = get_or_train_model(archive_file, model_path, n_training=500)
    
    n_colonies = 2
    params = {
        'max_iter': 30,
        'n_ants': 1500,
        'top_k': 50,
        'alpha': 1.0,
        'beta': 1.0,
    }
    
    futures = []
    results = []
    initial_lambdas_per_colony = []
    learner_tmp = ACOActiveLearner(archive_data, surrogate_model)
    learner_tmp.compute_heuristic()
    already_selected = set()
    for colony_id in range(n_colonies):
        learner_tmp.select_initial_diverse_lambdas(
            n_init_diverse=params['n_ants'],
            random_state=colony_id + GLOBAL_SEED,
            exclude_lambdas=already_selected
        )
        initial_lambdas = list(learner_tmp.Selected - already_selected)
        initial_lambdas_per_colony.append(initial_lambdas)
        already_selected.update(initial_lambdas)
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        for colony_id in range(n_colonies):
            futures.append(
                executor.submit(
                    run_colony,
                    colony_id,
                    archive_data,
                    surrogate_model,
                    C_ref,
                    params,
                    initial_lambdas_per_colony[colony_id]
                )
            )

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            print(f"[INFO] Colonia completata: {future.result()}")

    all_selected = []
    all_Ce = []
    all_error_lists = []
    for selected, C_e, error_list in results:
        all_selected.extend(selected)
        all_Ce.append(C_e)
        all_error_lists.append(error_list)
    all_selected_unique = [tuple(lam) for lam in {tuple(np.round(lam, 8)) for lam in all_selected}]
    print(f"\n[INFO] Lambda selezionati totali (unici): {len(all_selected_unique)}")
    print(all_selected_unique)



    C_e_union = learner_tmp.compute_Ce_from_lambdas(all_selected_unique)
    error_union = np.linalg.norm(C_ref - C_e_union, ord='fro') if C_e_union is not None else None

    print("\n--- RISULTATI FINALI MULTICOLONY ---")
    print(f"Numero lambda selezionati (unici): {len(all_selected_unique)}")
    print(f"Errore finale (unione): {error_union:.4f}")
    print(f"\nMatrice C_ref:\n{C_ref}")
    print(f"\nMatrice C_e_union:\n{C_e_union}")
    print("Norma Frobenius di C_e_union:", np.linalg.norm(C_ref - C_e_union, ord='fro'))
    print("Shape di C_e_union:", C_e_union.shape)
    
if __name__ == "__main__":
    main()