from surrogate_model import load_lambda_covariance_data,
from aco_active_learning import ACOInitializer
def main():
    # Carica dataset per surrogato
    archive_data = load_lambda_covariance_data('datasets/losses_cov.csv')

    # Genera un set di lambda di riferimento per stimare C_ref
    # Puoi usare un campione estratto dall'archive o generarne uno apposito
    lambda_ref_samples = archive_data[['lambda1', 'lambda2', 'lambda3']].values
    print(f"Calcolo C_ref su {len(lambda_ref_samples)} campioni lambda...")

    results_df = pd.read_csv('datasets/results.csv')
    C_ref = results_df.values

    print("Inizializzazione ACO con warm-up surrogato...")
    aco = initialize_aco(lambda_data_path=archive_file)

    print("Avvio ciclo iterativo ACO + Active Learning...")
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
        retrain_every=5
    )

    print(f"Final selected lambda vectors: {len(Selected)}")
    print(f"Final error ||C_ref - C_e||_F = {final_error:.6f}")

    # Visualizza la covarianza empirica finale
    plt.figure(figsize=(6, 5))
    plt.imshow(C_e, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Covariance Value')
    plt.title('Empirical Covariance Matrix C_e (Final)')
    plt.tight_layout()
    plt.savefig('final_covariance_matrix_Ce.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()