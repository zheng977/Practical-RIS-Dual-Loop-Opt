"""
Parameter sweep experiment: Compare the impact of different coupling degrees (alpha_min)
and different curve steepness (gamma) on MSE.
Only compares the true MSE values of coupled GA and uncoupled GA.
"""
import numpy as np
import matplotlib.pyplot as plt
from ..ao_algorithm import ao_algorithm
from ..optimizers import ga_optimize_omega, no_optimize_omega
from ..channel import channel_simulation
from ..location import location
import torch
import pandas as pd
import time
from datetime import datetime


def calculate_mse(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param):
    """
    Directly calculates the MSE value.
    """
    # Ensure eta is a real number, take absolute value if complex
    if isinstance(eta, complex):
        eta = abs(eta)

    # Calculate reflection coefficient alpha
    alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min

    # Construct RIS reflection matrix Φ
    Phi = np.diag(alpha * np.exp(1j * omega))

    # Calculate equivalent channel h_e
    h_e = g.conj().T @ Phi @ h

    # Calculate MSE (ensure the result is real)
    K = h.shape[1]
    mse_complex = np.sum(np.abs(eta * h_e * b - 1 / K) ** 2) + eta ** 2 * sigma_a2

    # If the result is still complex, take the real part
    if isinstance(mse_complex, complex):
        mse = np.real(mse_complex)
    else:
        mse = mse_complex

    return mse


def run_experiment(h, g, alpha_min_values, gamma_values, seed=2024):
    """
    Runs the parameter sweep experiment, comparing only coupled GA and uncoupled GA.

    Parameters:
    - h: Channel matrix from users to RIS
    - g: Channel vector from RIS to AP
    - alpha_min_values: Different coupling degree values
    - gamma_values: Different curve steepness values
    - seed: Random seed

    Returns:
    - results_df: DataFrame containing all experiment results
    """
    # Fixed parameters
    K = h.shape[1]  # Number of users
    N = h.shape[0]  # Number of RIS elements
    sigma_a2 = 1e-12  # AP noise variance (-110dB)
    max_iter = 50  # Maximum iterations
    tol = 1e-3  # Convergence threshold
    phi = 0.0  # Phase shift parameter
    P_k = np.ones(K) * 1  # Transmit power constraint for each user

    # GA parameters
    population_size = 80  # Population size
    num_generations = 100  # Number of generations
    crossover_rate = 0.8  # Crossover rate
    mutation_rate = 0.3  # Mutation rate

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize variables
    eta_init = 1.0
    b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
    omega_init = np.random.uniform(low=-1 * np.pi, high=1 * np.pi, size=N)

    # Prepare results storage
    results = []

    # Start parameter sweep experiment
    total_experiments = len(alpha_min_values) * len(gamma_values) * 2  # Coupled GA and Uncoupled GA
    experiment_count = 0

    start_time = time.time()

    # First, run uncoupled GA (alpha_min=1.0)
    print("\nRunning uncoupled GA (alpha_min=1.0)...")
    # Optimize omega using GA first
    omega_opt_uncoupled_ga = ga_optimize_omega(
        h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
        alpha_min=1.0, phi=phi, gamma_param=0.0,
        sigma_a2=sigma_a2, population_size=population_size,
        num_generations=num_generations, crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    # Use GA-optimized omega as initial value for AO algorithm
    eta_uncoupled_ga, b_uncoupled_ga, omega_uncoupled_ga, mse_list_uncoupled_ga = ao_algorithm(
        h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_opt_uncoupled_ga,
        P_k=P_k, sigma_a2=sigma_a2, alpha_min=1.0,
        phi=phi, gamma_param=0.0, optimize_omega_func=no_optimize_omega,
        max_iter=max_iter, tol=tol
    )

    # Record baseline MSE for uncoupled GA
    mse_uncoupled_ga = mse_list_uncoupled_ga[-1]
    results.append({
        'algorithm': 'Uncoupled_GA',
        'alpha_min': 1.0,
        'gamma': 0.0,
        'mse': mse_uncoupled_ga
    })

    # Iterate through different alpha_min and gamma values
    for alpha_min in alpha_min_values:
        for gamma in gamma_values:
            print(f"\nStarting experiment: alpha_min={alpha_min}, gamma={gamma}")

            # ========== Coupled GA Optimization ==========
            print(f"Running coupled GA optimization: alpha_min={alpha_min}, gamma={gamma}...")
            # Optimize omega using GA first
            omega_opt_ga = ga_optimize_omega(
                h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
                alpha_min=alpha_min, phi=phi, gamma_param=gamma,
                sigma_a2=sigma_a2, population_size=population_size,
                num_generations=num_generations, crossover_rate=crossover_rate,
                mutation_rate=mutation_rate
            )

            # Use GA-optimized omega as initial value for AO algorithm
            eta_opt_ga, b_opt_ga, omega_opt_ga, mse_list_ga = ao_algorithm(
                h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_opt_ga,
                P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min,
                phi=phi, gamma_param=gamma, optimize_omega_func=no_optimize_omega,
                max_iter=max_iter, tol=tol
            )

            # Record coupled GA results
            mse_ga = mse_list_ga[-1]
            results.append({
                'algorithm': 'Coupled_GA',
                'alpha_min': alpha_min,
                'gamma': gamma,
                'mse': mse_ga
            })

            experiment_count += 1
            print(f"Experiment progress: {experiment_count}/{total_experiments}, "
                  f"{experiment_count / total_experiments * 100:.1f}% complete")

            # Calculate MSE for uncoupled GA under current conditions
            mse_uncoupled_current = calculate_mse(
                h=h, g=g, eta=eta_uncoupled_ga, b=b_uncoupled_ga, omega=omega_uncoupled_ga,
                sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma
            )

            # Record MSE for uncoupled GA under current parameters
            results.append({
                'algorithm': 'Uncoupled_GA_Current',
                'alpha_min': alpha_min,
                'gamma': gamma,
                'mse': mse_uncoupled_current
            })

            experiment_count += 1
            print(f"Experiment progress: {experiment_count}/{total_experiments}, "
                  f"{experiment_count / total_experiments * 100:.1f}% complete")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExperiment complete! Total time: {elapsed_time:.2f} seconds")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def plot_results(results_df):
    """
    Plots the experiment results.
    """
    # Impact of different alpha_min values on MSE
    plt.figure(figsize=(15, 10))

    # Group and plot by gamma
    gamma_values = sorted(results_df['gamma'].unique())

    for i, gamma in enumerate(gamma_values):
        plt.subplot(2, 2, i + 1)

        # Filter data for the current gamma value
        df_gamma = results_df[
            (results_df['gamma'] == gamma) &
            (results_df['algorithm'].isin(['Coupled_GA', 'Uncoupled_GA_Current']))
            ]

        # Group by algorithm and alpha_min
        for algorithm in ['Coupled_GA', 'Uncoupled_GA_Current']:
            df_alg = df_gamma[df_gamma['algorithm'] == algorithm]
            plt.plot(df_alg['alpha_min'], df_alg['mse'],
                     marker='o',
                     label=algorithm)

        plt.title(f'gamma = {gamma}')
        plt.xlabel('alpha_min (Coupling Degree)')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('alpha_min_vs_mse.png')

    # Impact of different gamma values on MSE
    plt.figure(figsize=(15, 10))

    # Group and plot by alpha_min
    alpha_min_values = sorted(results_df['alpha_min'].unique())

    for i, alpha_min in enumerate(alpha_min_values):
        plt.subplot(2, 2, i + 1)

        # Filter data for the current alpha_min value
        df_alpha = results_df[
            (results_df['alpha_min'] == alpha_min) &
            (results_df['algorithm'].isin(['Coupled_GA', 'Uncoupled_GA_Current']))
            ]

        # Group by algorithm and gamma
        for algorithm in ['Coupled_GA', 'Uncoupled_GA_Current']:
            df_alg = df_alpha[df_alpha['algorithm'] == algorithm]
            plt.plot(df_alg['gamma'], df_alg['mse'],
                     marker='o',
                     label=algorithm)

        plt.title(f'alpha_min = {alpha_min}')
        plt.xlabel('gamma (Curve Steepness)')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('gamma_vs_mse.png')
    plt.show()


def main():
    """
    Main function to run the experiment and plot results.
    """
    # Simulation parameters
    K = 10  # Number of users
    N = 64  # Number of RIS elements
    M = 1  # Number of AP antennas (assuming SISO at AP for simplicity in g)

    # Generate channel
    pL_UA, pL_UR, pL_RA = location(K)
    _, hr, G = channel_simulation(pL_UA, pL_UR, pL_RA, K, N, M)

    # For simplicity, we assume g is a vector (M=1). If M > 1, g would be a matrix.
    g_vector = G[0, :]  # Take the first row if M > 1, or it's already a vector if M=1

    # Define parameter ranges
    alpha_min_values = [0.0, 0.2, 0.4, 0.6, 0.8]  # Different coupling degrees
    gamma_values = [0.5, 1.0, 1.5, 2.0]      # Different curve steepness values

    # Run the experiment
    results_df = run_experiment(hr, g_vector, alpha_min_values, gamma_values)

    # Plot results
    plot_results(results_df)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'comparison_results_{timestamp}.csv', index=False)
    print(f"\nResults saved to comparison_results_{timestamp}.csv")


if __name__ == "__main__":
    main()