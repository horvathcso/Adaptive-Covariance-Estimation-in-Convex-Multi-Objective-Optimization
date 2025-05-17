# aco_active_learning.py
import pandas as pd
import numpy as np
from collections import defaultdict
from surrogate_model import fit_gp_model, optimize_for_lambda, hessian_estimation_for_lambda, estimate_local_covariances_from_lambdas,load_lambda_covariance_data,train_and_prepare_surrogate
import matplotlib.pyplot as plt
import os
import joblib

NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations
GLOBAL_SEED = 42 # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)

class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None):
        self.lambda_data = lambda_data.copy()
        self.lambda_data['lambda3'] = 1 - self.lambda_data['lambda1'] - self.lambda_data['lambda2']
        self.lambdas = self.lambda_data[['lambda1', 'lambda2', 'lambda3']].values

        self.tau = defaultdict(lambda: 1.0)
        self.eta = {} # Heuristic values, norma
        self.surrogate = surrogate_model

        self.Selected = set()
        self.C_e = None

        self.epsilon_threshold = 1e-2

    def compute_heuristic(self):
        if self.surrogate is None:
            raise ValueError("Surrogate model non definito.")
        # Calcola la norma di sensibilità per ogni lambda , deve usare il modello surrogato
        print("Calcolo euristica per ogni lambda...")
        X = self.lambda_data[['lambda1', 'lambda2']].values
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        for i, lam in enumerate(self.lambdas):
            self.eta[tuple(lam)] = y_pred[i]
            print(f"Lambda: {lam}, Heuristic (eta): {y_pred[i]}")

    # Restituisce i valori di tau e eta
    def get_pheromone_and_heuristic(self):
        return self.tau, self.eta

    def get_initial_state(self):
        return self.Selected, self.C_e, self.epsilon_threshold

    def select_initial_diverse_lambdas(self, n_init_diverse=5):
        """
        Seleziona n_init_diverse lambda iniziali massimizzando la diversità (distanza euclidea).
        """
        from scipy.spatial.distance import cdist

        all_lambdas = self.lambdas.copy()
        selected = []
        # Scegli il primo punto casualmente
        idx = np.random.choice(len(all_lambdas))
        selected.append(all_lambdas[idx])

        for _ in range(1, n_init_diverse):
            # Calcola la distanza minima di ogni candidato dai già selezionati
            dists = cdist(all_lambdas, np.array(selected))
            min_dists = dists.min(axis=1)
            # Escludi già selezionati
            for s in selected:
                mask = np.all(all_lambdas == s, axis=1)
                min_dists[mask] = -np.inf
            # Scegli il candidato con la massima distanza minima
            idx = np.argmax(min_dists)
            selected.append(all_lambdas[idx])

        # Aggiorna self.Selected
        for lam in selected:
            self.Selected.add(tuple(lam))
        print(f"Lambda iniziali selezionati (diversi): {selected}")

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
        if len(self.Selected) <= 1:
            print("Numero insufficiente di elementi selezionati per calcolare C_e.")
            self.C_e = None
            return

        x_list = []
        for lam in self.Selected:
            _, x_opt = hessian_estimation_for_lambda(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar

        # Calcola C_e solo se ci sono abbastanza elementi
        if len(self.Selected) > 1:
            self.C_e = (deviations.T @ deviations) / (len(self.Selected) - 1)
        else:
            self.C_e = None

    def run_aco_active_learning(self, C_ref, archive_data, n_ants=20, top_k=5,
                                alpha=1.0, beta=1.0, omega=0.7,
                                epsilon=None, budget=50, retrain_every=5,
                                n_init_diverse=5):
        if epsilon is None:
            epsilon = self.epsilon_threshold

        # Selezione iniziale diversificata
        if len(self.Selected) == 0:
            self.select_initial_diverse_lambdas(n_init_diverse=n_init_diverse)
            # Aggiorna C_e dopo la selezione iniziale
            self.update_selected_and_Ce(next(iter(self.Selected)))

        val_count = len(self.Selected)
        while val_count < budget:
            print(f"Iterazione {val_count + 1}/{budget}")
            candidates = self.sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)

            for lam in candidates:
                X = np.array([[lam[0], lam[1]]])
                X_scaled = self.surrogate['scaler_X'].transform(X)
                y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
                y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                self.eta[lam] = y_pred[0]
                print(f"Lambda: {lam}, Predicted Heuristic: {y_pred[0]}")

            delta_k_list = []
            for lam in candidates:
                delta_k = self.compute_delta_k(C_ref, lam)
                delta_k_list.append((lam, delta_k))

            scored_candidates = []
            for lam, delta_k in delta_k_list:
                score = omega * delta_k + (1 - omega) * self.eta[lam]
                scored_candidates.append((lam, score))
                print(f"Lambda: {lam}, Score: {score}")

            scored_candidates.sort(key=lambda x: x[1])
            top_candidates = [lam for lam, _ in scored_candidates[:top_k]]
            print(f"Top-k candidati: {top_candidates}")

            for lam in top_candidates:
                _, x_opt = optimize_for_lambda(lam)
                cov_matrix, _ = hessian_estimation_for_lambda(lam)
                sens_norm = np.linalg.norm(cov_matrix, ord='fro')

                self.update_selected_and_Ce(lam, cov_matrix)

                self.tau[lam] += sens_norm
                self.eta[lam] = sens_norm
                print(f"Lambda selezionato: {lam}, Sensitivity Norm: {sens_norm}")

                val_count += 1

                if self.C_e is not None:
                    if C_ref.shape != self.C_e.shape:
                        raise ValueError(f"Dimensioni incompatibili: C_ref {C_ref.shape}, C_e {self.C_e.shape}")

                    error = np.linalg.norm(C_ref - self.C_e, ord='fro')
                    print(f"Errore attuale: {error}")
                    if error < epsilon or val_count >= budget:
                        print("Condizione di arresto raggiunta.")
                        return list(self.Selected), self.C_e, error

            if val_count % retrain_every == 0:
                print("Riaddestramento del modello surrogato...")
                self.retrain_surrogate(archive_data)

        final_error = np.linalg.norm(C_ref - self.C_e, ord='fro') if self.C_e is not None else None
        print(f"Errore finale: {final_error}")
        return list(self.Selected), self.C_e, final_error
    # Calcola la variazione dell'errore dall'aggiunta del nuovo lambda
    def compute_delta_k(self, C_ref, candidate_lambda):
        print(f"Calcolo delta_k per lambda: {candidate_lambda}")
        selected_temp = self.Selected.union({tuple(candidate_lambda)})
        if len(selected_temp) <= 1:
            print("Numero insufficiente di elementi selezionati per calcolare Ce_augmented.")
            return np.nan  # Corretto qui
        x_list = []
        for lam in selected_temp:
            _, x_opt = hessian_estimation_for_lambda(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        Ce_augmented = (deviations.T @ deviations) / (len(selected_temp) - 1)
        if C_ref.shape != Ce_augmented.shape:
            raise ValueError(f"Dimensioni incompatibili: C_ref {C_ref.shape}, Ce_augmented {Ce_augmented.shape}")

        Ce_current = self.C_e if self.C_e is not None else np.zeros_like(C_ref)
        delta_k = np.linalg.norm(C_ref - Ce_augmented, ord='fro') - np.linalg.norm(C_ref - Ce_current, ord='fro')
        print(f"Delta_k: {delta_k}")
        return delta_k

    def retrain_surrogate(self, archive_data):
        """
        Riaddestra il surrogato S(λ) usando dati archivio più Selected.
        archive_data: DataFrame contenente dati iniziali + valutazioni esatte
        """

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

        # Ricalcola l'euristica
        self.compute_heuristic()


    # Funzione principale per l'ACO + Active Learning
    def run_aco_active_learning(self, C_ref, archive_data, n_ants=20, top_k=5,
                               alpha=1.0, beta=1.0, omega=0.7,
                               epsilon=None, budget=50, retrain_every=5, n_init_diverse=5):
        if epsilon is None:
            epsilon = self.epsilon_threshold

        val_count = 0
        # fissa iterazioni per il ciclo
        while val_count < budget:
            print(f"Iterazione {val_count + 1}/{budget}")
            # 1. Campionamento candidati ACO
            candidates = self.sample_candidates(n_ants=n_ants, alpha=alpha, beta=beta)

            # 2. Ricalcolo euristica per candidati
            for lam in candidates:
                X = np.array([[lam[0], lam[1]]])
                X_scaled = self.surrogate['scaler_X'].transform(X)
                y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
                y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                self.eta[lam] = y_pred[0]
                print(f"Lambda: {lam}, Predicted Heuristic: {y_pred[0]}")

            # 3. Calcolo Δ_k
            delta_k_list = []
            for lam in candidates:
                delta_k = self.compute_delta_k(C_ref, lam)
                delta_k_list.append((lam, delta_k))

            # 4. Selezione top-k (mix Δ_k e η)
            scored_candidates = []
            for lam, delta_k in delta_k_list:
                score = omega * delta_k + (1 - omega) * self.eta[lam]
                scored_candidates.append((lam, score))
                print(f"Lambda: {lam}, Score: {score}")

            scored_candidates.sort(key=lambda x: x[1])  # score più basso = maggiore riduzione errore
            top_candidates = [lam for lam, _ in scored_candidates[:top_k]]
            print(f"Top-k candidati: {top_candidates}")

            # 5. Valutazione esatta e aggiornamento
            for lam in top_candidates:
                _, x_opt = optimize_for_lambda(lam)
                cov_matrix, _ = hessian_estimation_for_lambda(lam)
                sens_norm = np.linalg.norm(cov_matrix, ord='fro')

                self.update_selected_and_Ce(lam, cov_matrix)

                # Update pheromone and heuristic
                self.tau[lam] += sens_norm
                self.eta[lam] = sens_norm
                print(f"Lambda selezionato: {lam}, Sensitivity Norm: {sens_norm}")

                val_count += 1

                # Controllo convergenza
                if self.C_e is not None:
                    error = np.linalg.norm(C_ref - self.C_e, ord='fro')
                    print(f"Errore attuale: {error}")
                    if error < epsilon or val_count >= budget:
                        print("Condizione di arresto raggiunta.")
                        return list(self.Selected), self.C_e, error

            # 6. Retraining surrogato periodico
            if val_count % retrain_every == 0:
                print("Riaddestramento del modello surrogato...")
                self.retrain_surrogate(archive_data)

        # Fine ciclo: ritorna stato finale
        final_error = np.linalg.norm(C_ref - self.C_e, ord='fro') if self.C_e is not None else None
        print(f"Errore finale: {final_error}")
        return list(self.Selected), self.C_e, final_error

    def get_or_train_model(archive_data, model_path='surrogate_model.pkl'):
        if os.path.exists(model_path):
            print("Caricamento del modello salvato...")
            gp_model = joblib.load(model_path)
        else:
            print("Modello non trovato. Eseguo il training...")
            gp_model = train_and_prepare_surrogate(archive_data, n_training=500)
            joblib.dump(gp_model, model_path)
            print("Modello salvato in:", model_path)
        return gp_model
# main.py

def main():
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance.csv'
    model_path = 'surrogate_model.pkl'

    # Carica il dataset di archivio
    archive_data=pd.read_csv(archive_file)
    # archive_data = load_lambda_covariance_data(archive_file)
    print(f"Caricati {len(archive_data)} campioni da '{archive_file}'")

    if os.path.exists(model_path):
        print("Caricamento del modello salvato...")
        gp_model = joblib.load(model_path)
    else:
        print("Modello non trovato. Eseguo il training...")
        gp_model = train_and_prepare_surrogate(archive_data, n_training=500)
        joblib.dump(gp_model, model_path)
        print("Modello salvato in:", model_path)

    # Carica i dati di ground truth
    results_df = pd.read_csv(ground_truth_file, header=None)
    C_ref = results_df.values
    print("Shape di C_ref:", C_ref.shape)
    # Inizializza l'archivio
    aco = ACOActiveLearner(archive_data, gp_model)
    # Fit iniziale surrogato (deve essere presente nel dataset)
    aco.compute_heuristic()
    Selected, C_e, final_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=30,
        top_k=5,
        alpha=1.0,
        beta=1.0,
        omega=0.7,
        epsilon=1e-2,
        budget=50,
        retrain_every=5,
        n_init_diverse=7  # ad esempio
    )

    print(f"Selected points: {len(Selected)}")
    print(f"Final error: {final_error:.6f}")

    plt.imshow(C_e, cmap='viridis')
    plt.colorbar()
    plt.title('Covarianza empirica finale C_e')
    plt.show()

if __name__ == "__main__":
    main()