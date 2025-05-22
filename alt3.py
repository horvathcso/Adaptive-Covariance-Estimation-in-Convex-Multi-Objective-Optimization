# aco_active_learning.py
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from surrogate_model import fit_gp_model, optimize_for_lambda, hessian_estimation_for_lambda, \
    estimate_local_covariances_from_lambdas, load_lambda_covariance_data, train_and_prepare_surrogate
import matplotlib.pyplot as plt
import os
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
import concurrent.futures
from filelock import FileLock
import time

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

NUM_PERTURBATIONS = 20
PERTURBATION_STRENGTH = 0.025


class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None, rho=0.1):
        self.lambda_data = lambda_data.copy()
        self.lambda_data['lambda3'] = 1 - self.lambda_data['lambda1'] - self.lambda_data['lambda2']
        self.lambdas = self.lambda_data[['lambda1', 'lambda2', 'lambda3']].values

        self.tau = defaultdict(lambda: 1.0)
        self.eta = {}
        self.surrogate = surrogate_model

        self.Selected = set()
        self.C_e = None

        self.epsilon_threshold = 1e-2
        self.rho = rho
        self.x_opt_cache = {}  # Cache per i risultati di optimize_for_lambda

    def get_x_opt(self, lam):
        lam_tuple = tuple(lam)
        if lam_tuple not in self.x_opt_cache:
            x_opt, _ = optimize_for_lambda(lam_tuple)
            self.x_opt_cache[lam_tuple] = x_opt
        return self.x_opt_cache[lam_tuple]

    def compute_heuristic(self):
        if self.surrogate is None:
            raise ValueError("Surrogate model non definito.")
        print("Calcolo eucristica per ogni lambda...")
        X = self.lambda_data[['lambda1', 'lambda2']].values
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred_mean = y_pred.mean()
        y_pred_std = y_pred.std() if y_pred.std() > 0 else 1.0  # evita divisione per zero
        for i, lam in enumerate(self.lambdas):
            lam_tuple = tuple(lam)
            # Penalità di diversità più stabile
            diversity_penalty = 1 / (1 + min([np.linalg.norm(lam - np.array(sel)) for sel in self.Selected] + [1.0]))
            # Normalizza la predizione e applica la penalità
            self.eta[lam_tuple] = ((y_pred[i] - y_pred_mean) / y_pred_std) * diversity_penalty

    def select_diverse_lambdas(self, n_select=20, min_dist=0.08, score_dict=None, exclude_lambdas=None,
                               random_state=None):
        print("Selezione lambda diversificati...")
        if score_dict is None:
            score_dict = self.eta
        if exclude_lambdas is None:
            exclude_lambdas = set()
        sorted_lambdas = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(sorted_lambdas)
        selected = []
        for lam, score in sorted_lambdas:
            lam_tuple = tuple(lam)
            if lam_tuple in selected or lam_tuple in exclude_lambdas:
                continue
            lam_arr = np.array(lam_tuple)
            if selected:
                dists = np.linalg.norm(np.array(selected) - lam_arr, axis=1)
                if np.all(dists >= min_dist):
                    selected.append(lam_tuple)
            else:
                selected.append(lam_tuple)
            if len(selected) >= n_select:
                break
        # Se non hai ancora raggiunto n_select, aggiungi i migliori rimanenti (anche se vicini)
        if len(selected) < n_select:
            for lam, score in sorted_lambdas:
                lam_tuple = tuple(lam)
                if lam_tuple not in selected and lam_tuple not in exclude_lambdas:
                    selected.append(lam_tuple)
                if len(selected) >= n_select:
                    break
        print(f"Numero di lambda dopo il pruning: {len(selected)}")
        return [tuple(lam) for lam in selected]

    def select_initial_diverse_lambdas(self, n_init_diverse=20, random_state=None):
        print("Selezione iniziale lambda diversificati...")
        sorted_lambdas = sorted(self.eta.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(sorted_lambdas)
        selected = []
        for lam, score in sorted_lambdas:
            lam_tuple = tuple(lam)
            if lam_tuple in selected:
                continue
            selected.append(lam_tuple)
            if len(selected) >= n_init_diverse:
                break
        print(f"Numero di lambda iniziali selezionati: {len(selected)}")
        return selected

    def sample_candidates(self, n_ants=100, alpha=1.0, beta=1.0, random_state=None):
        print("Campionamento candidati...")
        keys = list(self.eta.keys())
        scores = np.array([
            (self.tau[lam] ** alpha) * (self.eta[lam] ** beta) for lam in keys
        ])
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
        n_ants = min(n_ants, len(keys))  # Fix: non più di len(keys) senza replacement
        indices = rng.choice(len(keys), size=n_ants, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        print(f"Candidati campionati: {len(sampled)}")
        return sampled

    def run_aco_active_learning(self, C_ref, archive_data, n_ants=150,
                                alpha=1.0, beta=1.0,
                                epsilon=None, budget=10, retrain_every=6,
                                n_init_diverse=5, exclude_lambdas=None, random_state=None,
                                min_lambda=10, top_k=20):
        if epsilon is None:
            epsilon = 1e-6
        if len(self.Selected) == 0:
            self.Selected = set(self.select_diverse_lambdas(n_select=n_init_diverse, min_dist=0.08))
            self.C_e = self.compute_Ce_from_lambdas(self.Selected)
        val_count = len(self.Selected)
        print(f"Numero iniziale di lambda selezionati: {val_count}", flush=True)
        best_error = float('inf')
        best_config = None
        error_list = []
        max_iter = budget
        for iter_idx in range(max_iter):
            print(f"\n--- Iterazione {iter_idx + 1}/{max_iter} ---", flush=True)
            print(f"Lambda selezionati all'inizio iterazione: {len(self.Selected)}")

            # 1. Campiona nuovi candidati dalla popolazione globale (non solo dai selezionati)
            candidates = self.sample_candidates(n_ants=top_k, alpha=alpha, beta=beta, random_state=random_state)
            # Escludi quelli già selezionati
            new_candidates = [lam for lam in candidates if lam not in self.Selected]
            print(f"Candidati nuovi per update: {len(new_candidates)}")

            # 2. Calcola delta_k per i nuovi candidati
            delta_k_list = []
            for lam in new_candidates:
                delta_k = self.compute_delta_k(C_ref, lam)
                delta_k_list.append((lam, delta_k))
            # Ordina per miglioramento
            delta_k_list.sort(key=lambda x: x[1])
            selected_candidates = [lam for lam, _ in delta_k_list[:max(1, len(delta_k_list) // 2)]]
            print(f"Candidati selezionati per update: {len(selected_candidates)}")

            self.evaporate_pheromones()
            self.update_pheromones(selected_candidates, C_ref)

            # 3. Aggiungi solo i nuovi lambda selezionati
            n_added = 0
            for lam in selected_candidates:
                lam_tuple = tuple(lam)
                if lam_tuple not in self.Selected:
                    self.Selected.add(lam_tuple)
                    n_added += 1
            print(f"Nuovi lambda aggiunti: {n_added}")
            print(f"Totale lambda dopo aggiunta: {len(self.Selected)}")

            # 4. Pruning come prima
            max_lambdas = 90  # valore più basso per forzare il pruning
            if len(self.Selected) > max_lambdas:
                before_pruning = set(self.Selected)
                self.Selected = set(
                    self.select_diverse_lambdas(n_select=max_lambdas, min_dist=0.08, score_dict=self.tau))
                n_removed = len(before_pruning) - len(self.Selected)
                print(f"Lambda rimossi dal pruning: {n_removed}")
                print(f"Lambda selezionati dopo il pruning: {len(self.Selected)}")
            else:
                print("Nessun pruning necessario, mantengo tutti i lambda selezionati.")

            self.C_e = self.compute_Ce_from_lambdas(self.Selected)
            if self.C_e is not None:
                error = np.linalg.norm(C_ref - self.C_e, ord='fro')
            else:
                error = None
            error_list.append(error)
            # Salva la configurazione migliore
            if error is not None and error < best_error:
                best_error = error
                best_config = list(self.Selected)
                print(
                    f"[Iterazione {iter_idx + 1}] Nuova configurazione migliore trovata con errore {best_error:.8f} e {len(best_config)} lambda.")
            else:
                print(
                    f"[Iterazione {iter_idx + 1}] Nessun miglioramento: errore attuale {error:.8f}, errore migliore {best_error:.8f}")
            print(f"Errore attuale dopo iterazione: {error:.8f}" if error is not None else "Errore attuale: N/A")
            if (error is not None and error < epsilon) or (len(self.Selected) < min_lambda):
                print(f"STOP: errore < {epsilon} o numero campioni < {min_lambda}")
                break

        plt.figure(figsize=(8, 4))
        plt.plot(error_list, marker='o')
        plt.title("Errore Frobenius tra $C_{ref}$ e $C_e$ per iterazione")
        plt.xlabel("Iterazione")
        plt.ylabel("Errore Frobenius")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("errore_per_iterazione.png")
        plt.show()
        return list(self.Selected), self.C_e, error_list[-1] if error_list else None, best_config, best_error

    def compute_Ce_from_lambdas(self, lambdas_list):
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            x_opt = self.get_x_opt(lam)
            x_list.append(x_opt)
        X = np.array(x_list)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        C_e = (deviations.T @ deviations) / (len(lambdas_list) - 1)
        return C_e

    def compute_delta_k(self, C_ref, lam):
        lam_tuple = tuple(lam)
        if not self.Selected:
            C_e_single = self.compute_Ce_from_lambdas([lam_tuple])
            delta_k = -np.sum(np.abs(C_ref - C_e_single))  # norma L1
        else:
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected) + [lam_tuple])
            delta_k = np.sum(np.abs(C_ref - C_e_new)) - np.sum(np.abs(C_ref - C_e_old))
        return delta_k

    def evaporate_pheromones(self):
        for lam in self.tau:
            self.tau[lam] *= (1 - self.rho)

    def update_pheromones(self, selected_lambdas, C_ref):
        for lam in selected_lambdas:
            lam_tuple = tuple(lam)
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected - {lam_tuple}))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected))
            if C_e_old is not None and C_e_new is not None:
                error_old = np.linalg.norm(C_ref - C_e_old, ord='fro')
                error_new = np.linalg.norm(C_ref - C_e_new, ord='fro')
                delta_error = error_old - error_new
                if delta_error > 0:
                    self.tau[lam_tuple] += delta_error


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


def run_single_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None, random_state=None):
    print(f"Avvio colonia {colony_id}")
    aco = ACOActiveLearner(archive_data, gp_model)
    print(f"Colonia {colony_id}: calcolo euristica")
    aco.compute_heuristic()
    print(f"Colonia {colony_id}: selezione iniziale diversificata")
    if already_selected is None:
        already_selected = set()
    aco.Selected = set(aco.select_diverse_lambdas(
        n_select=params['n_init_diverse'],
        min_dist=0.08,
        exclude_lambdas=already_selected,
        random_state=random_state  # <--- aggiungi questo parametro!
    ))
    print(f"Colonia {colony_id}: avvio ACO, lambda iniziali :{list(aco.Selected)}")

    Selected, C_e, final_error, best_config, best_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=params['n_ants'],
        alpha=params['alpha'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        retrain_every=params['retrain_every'],
        n_init_diverse=params['n_init_diverse'],
        exclude_lambdas=already_selected,
        random_state=random_state,
        min_lambda=params['n_init_diverse'] // 2,
        top_k=params['top_k']
    )
    return {
        "selected": list(Selected),
        "best_config": best_config,
        "best_error": best_error
    }


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


# main.py

def main():
    print("Inizio script alternativa 3", flush=True)
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    print(f"Caricati {len(archive_data)} campioni da '{archive_file}'", flush=True)

    gp_model = get_or_train_model(archive_file, model_path, n_training=500)

    results_df = pd.read_csv(ground_truth_file, header=None)
    C_ref = results_df.values
    print(f"Shape di C_ref: {C_ref.shape}", flush=True)

    params = dict(
        n_ants=150,
        top_k=50,
        alpha=1.0,
        beta=1.0,
        omega=0.7,
        epsilon=0.1,
        budget=10,
        retrain_every=5,
        n_init_diverse=50
    )

    n_colonies = 2
    all_best_configs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        futures = [
            executor.submit(
                run_single_colony,
                i,
                archive_data,
                gp_model,
                C_ref,
                params,
                random_state=GLOBAL_SEED + i
            )
            for i in range(n_colonies)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_best_configs.append(result["best_config"])

    # Unione dei migliori lambda di tutte le colonie
    all_selected_unique = list({tuple(lam): lam for config in all_best_configs for lam in config}.values())
    print(f"Totale lambda selezionati (unici, migliori): {len(all_selected_unique)}")
    C_e_final = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(all_selected_unique)
    final_error = np.linalg.norm(C_ref - C_e_final, ord='fro') if C_e_final is not None else None
    print(f"Errore finale combinato (solo migliori): {final_error}")

    print("\nMatrice C_ref (Ground Truth):")
    print(np.array2string(C_ref, precision=4, suppress_small=True))

    print("\nMatrice C_e_final (Empirica):")
    if C_e_final is not None:
        print(np.array2string(C_e_final, precision=4, suppress_small=True))
    else:
        print("C_e_final non disponibile.")

    colony_errors = []
    for i in range(n_colonies):
        colony_selected = futures[i].result()["selected"]
        C_e_colony = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(colony_selected)
        err = np.linalg.norm(C_ref - C_e_colony, ord='fro') if C_e_colony is not None else None
        colony_errors.append(err)

    plt.figure(figsize=(6, 4))
    plt.bar(range(n_colonies), colony_errors)
    plt.xlabel("Colonia")
    plt.ylabel("Errore Frobenius finale")
    plt.title("Errore finale per colonia")
    plt.tight_layout()
    plt.savefig("final_error_per_colony.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(C_ref, aspect='auto', cmap='viridis')
    plt.title("C_ref (Ground Truth)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
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
    print(f"Errore finale (unione): {final_error:.4f}")
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