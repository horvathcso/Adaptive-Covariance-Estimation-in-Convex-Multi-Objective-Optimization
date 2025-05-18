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

    def compute_Ce_from_lambdas(lambdas_list):
        if len(lambdas_list) <= 1:
            return np.zeros_like(C_ref)
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



def proxy_cov_diff( C_ref, lam):
    x_list = []
    expected_shape = None
    for l in Selected:
        _, x_opt = optimize_for_lambda(l)
        x_opt_flat = np.asarray(x_opt).flatten()
        if expected_shape is None:
            expected_shape = x_opt_flat.shape
        if x_opt_flat.shape != expected_shape:
            # Pad o tronca per forzare la shape
            x_opt_flat = np.resize(x_opt_flat, expected_shape)
        x_list.append(x_opt_flat)
    _, x_opt_new = optimize_for_lambda(lam)
    x_opt_new_flat = np.asarray(x_opt_new).flatten()
    if expected_shape is not None and x_opt_new_flat.shape != expected_shape:
        x_opt_new_flat = np.resize(x_opt_new_flat, expected_shape)
    x_list.append(x_opt_new_flat)
    X = np.vstack(x_list)
    x_bar = np.mean(X, axis=0)
    deviations = X - x_bar
    if len(x_list) > 1:
        C_e_proxy = (deviations.T @ deviations) / (len(x_list) - 1)
    else:
        C_e_proxy = np.zeros((X.shape[1], X.shape[1]))
    return np.linalg.norm(C_ref - C_e_proxy, ord='fro')

def fitness_moaco(lam, norm, C_ref, Selected, w1, w2):
    S_lam = norm[lam]
    C_e_proxy = compute_Ce_from_lambdas(list(Selected) + [lam])
    proxy_diff = np.linalg.norm(C_ref - C_e_proxy, ord='fro')
    return w1 * S_lam - w2 * proxy_diff

def optimize_weights(C_ref, Selected, candidates, norm, w_grid):
    best_score = np.inf
    best_weights = None
    for w1 in w_grid['w1']:
        for w2 in w_grid['w2']:
            fitness_list = [(lam, fitness_moaco(lam, norm, C_ref, Selected, w1, w2)) for lam in candidates]
            fitness_list.sort(key=lambda x: x[1], reverse=True)
            top_k = [lam for lam, _ in fitness_list[:5]]
            delta_sum = 0
            for lam in top_k:
                C_e_old = compute_Ce_from_lambdas(Selected)
                C_e_new = compute_Ce_from_lambdas(list(Selected) + [lam])
                delta_k = np.linalg.norm(C_ref - C_e_new, ord='fro') - np.linalg.norm(C_ref - C_e_old, ord='fro')
                delta_sum += abs(delta_k)
            if delta_sum < best_score:
                best_score = delta_sum
                best_weights = (w1, w2)
    return best_weights



def main():
    print("Inizio script MOACO logica pesata")
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    C_ref = pd.read_csv(ground_truth_file, header=None).values

    surrogate_model = get_or_train_model(archive_file, model_path, n_training=500)

    # 1. Inizializzazione
    Lambda = [tuple(row) for row in archive_data[['lambda1', 'lambda2', 'lambda3']].values]
    Selected = set()
    C_e = np.zeros_like(C_ref)
    tau = defaultdict(lambda: 1.0)
    norm = {}
    w1, w2 = 1.0, 1.0

    n_ants = 30
    top_k = 5
    retrain_every = 5
    budget = 30
    alpha = 1.0
    beta = 1.0

    error_list = []
    w_grid = {'w1': np.linspace(0.5, 2.0, 4), 'w2': np.linspace(0.5, 2.0, 4)}

    val_count = len(Selected)
    iter_num = 0

    while val_count < budget:
        print(f"\nIterazione {val_count+1}/{budget}")

        # 2. Ottimizzazione pesi
        # Aggiorna norm per tutti i candidati
        for lam in Lambda:
            X = np.array([[lam[0], lam[1]]])
            X_scaled = surrogate_model['scaler_X'].transform(X)
            y_pred_scaled, _ = surrogate_model['model'].predict(X_scaled, return_std=True)
            y_pred = surrogate_model['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            norm[lam] = y_pred[0]
        candidates = sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)
        best_weights = optimize_weights(C_ref, Selected, candidates, norm, w_grid)
        if best_weights is not None:
            w1, w2 = best_weights
        print(f"Pesi ottimali trovati: w1={w1}, w2={w2}")

        # 3. Loop ACO
        C_e_current = compute_Ce_from_lambdas(Selected)
        fitness_list = [(lam, fitness_moaco(lam, norm, C_ref, Selected, w1, w2)) for lam in candidates]
        fitness_list.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [lam for lam, _ in fitness_list[:top_k]]

        print("Top-k candidati:")
        for lam, fit in fitness_list[:top_k]:
            print(f"  Lambda: {lam}, Fitness: {fit:.4f}")

        for lam in top_candidates:
            if lam not in Selected:
                _, x_opt = optimize_for_lambda(lam)
                cov_matrix = compute_covariance_matrix(lam)
                sens_norm = np.linalg.norm(cov_matrix, ord='fro')
                Selected.add(lam)
                tau[lam] += sens_norm
                norm[lam] = sens_norm

        # Aggiorna C_e
        C_e = compute_Ce_from_lambdas(Selected)
        error = np.linalg.norm(C_ref - C_e, ord='fro')
        error_list.append(error)
        print(f"Errore attuale dopo batch: {error:.4f}")

        # Aggiorna surrogato ogni retrain_every iterazioni
        iter_num += 1
        if iter_num % retrain_every == 0:
            # Riaddestra il surrogato con i nuovi dati
            # (implementa se vuoi, oppure lascia commentato)
            pass

        val_count = len(Selected)

        # Condizione di terminazione
        if error < 1e-3:
            print("Errore sufficientemente piccolo, termino.")
            break

    print("\n==== RISULTATI FINALI ====")
    print(f"Selected points: {len(Selected)}")
    print(f"Final error: {error:.6f}")
    print(f"Pesi ottimali finali: w1={w1}, w2={w2}")
    for lam in Selected:
        print(f"  {lam}")

    plt.plot(error_list)
    plt.xlabel("Iterazione")
    plt.ylabel("Errore Frobenius")
    plt.title("Andamento dell'errore per iterazione")
    plt.grid(True)
    plt.savefig("errore_moaco.png")
    print("Plot salvato come 'errore_moaco.png'")

if __name__ == "__main__":
    main()