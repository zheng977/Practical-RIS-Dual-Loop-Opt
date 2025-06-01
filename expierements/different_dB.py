# -*- coding: utf-8 -*-
"""
Main program to execute the Alternating Optimization (AO) algorithm,
comparing the effectiveness of different optimization methods (e.g., SCA, GA, PSO, no optimization)
under varying numbers of RIS elements N.
"""

import numpy as np
import matplotlib.pyplot as plt
# from channel_simulation import generate_channels_new # Original, likely needs adjustment or removal
from ..ao_algorithm import ao_algorithm # Adjusted import
from ..optimizers import ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca # Adjusted import
from ..channel import channel_simulation # Adjusted import
from ..location import location # Adjusted import

# Set Chinese font to avoid Matplotlib font warnings (can be removed if not needed)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# Set random seed for reproducibility
np.random.seed(2024)

def calculate_mse(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param):
    """
    Directly calculates the MSE value.

    Parameters:
    - h: Channel matrix from users to RIS (N, K)
    - g: Channel vector from RIS to AP (N,)
    - eta: Optimized eta value
    - b: Optimized b vector (K,)
    - omega: Optimized omega vector (N,)
    - sigma_a2: Noise variance
    - alpha_min: Minimum reflection coefficient of RIS
    - phi: Phase shift
    - gamma_param: Gamma parameter

    Returns:
    - mse: Calculated MSE value
    """
    # Calculate reflection coefficient alpha
    alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min

    # Construct RIS reflection matrix Φ
    Phi = np.diag(alpha * np.exp(1j * omega))

    # Calculate equivalent channel h_e
    h_e = g.conj().T @ Phi @ h

    # Calculate MSE
    K = h.shape[1]
    mse = np.sum(np.abs(eta * h_e * b - 1 / K) ** 2) + eta ** 2 * sigma_a2

    return mse

def run_experiment(sigma_a2_dB_list, K, N, max_iter, tol, alpha_min, phi, gamma_param, P_k):
    """
    Runs experiments at different noise power levels (dB) and compares MSE for different optimization methods.

    Parameters:
    - sigma_a2_dB_list: List of noise power levels in dB (e.g., [-120, -110, ..., -40])
    - K: Number of users
    - N: Number of RIS elements (fixed at 200 in the original script)
    - max_iter: Maximum AO iterations
    - tol: Convergence threshold
    - alpha_min: Minimum RIS reflection coefficient
    - phi: Phase shift parameter
    - gamma_param: Gamma parameter
    - P_k: Transmit power constraint for each user

    Returns:
    - mse_results: Dictionary, keys are optimization methods ('GA', 'Only Optimize eta and b', 'No Optimization at All'),
                   values are lists of MSEs for each noise power level.
    """
    mse_results = {
        'GA': [],
        'Only Optimize eta and b': [],
        'No Optimization at All': []
    }

    for sigma_a2_dB in sigma_a2_dB_list:
        print(f"\nRunning experiment with noise power σ_a^2 = {sigma_a2_dB} dB...")
        # Convert dB to linear value
        sigma_a2 = 10 ** (sigma_a2_dB / 10)

        # Generate channels (N is fixed based on parameter)
        pL_UA, pL_UR, pL_RA = location(K)
        # location_AP = np.array([-50, 0, 12]) # Defined in location.py
        # location_RIS = np.array([0, 0, 12]) # Defined in location.py
        _, h, g = channel_simulation(pL_UA, pL_UR, pL_RA, K, N, 1) # Assuming M=1 for AP antennas
        g = g.squeeze() # Ensure g is a vector

        print(f"Channel matrix h shape: {h.shape}")
        print(f"Channel vector g shape: {g.shape}")

        # Initialize variables
        eta_init = 1.0
        b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
        omega_init = np.random.uniform(low=-np.pi, high=np.pi, size=N)

        # No optimization at all: Directly calculate MSE with initial values
        print("Calculating MSE with no optimization (using initial values)...")
        mse_no_opt_all = calculate_mse(
            h=h, g=g, eta=eta_init, b=b_init, omega=omega_init,
            sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param
        )
        print(f"MSE with no optimization: {mse_no_opt_all}")
        mse_results['No Optimization at All'].append(mse_no_opt_all)

        # GA optimization
        print("Running AO algorithm with omega_n optimized by GA...")
        omega_opt_ga = ga_optimize_omega(
            h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
            alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
            sigma_a2=sigma_a2, population_size=80, num_generations=100,
            crossover_rate=0.8, mutation_rate=0.3
        )

        print("Performing AO optimization using the best phase found by GA...")
        _, _, _, mse_list_ga = ao_algorithm(
            h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_opt_ga,
            P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
            gamma_param=gamma_param, optimize_omega_func=no_optimize_omega, # Omega is fixed after GA
            max_iter=max_iter, tol=tol
        )
        mse_ga = mse_list_ga[-1]
        print(f"Final MSE with GA optimization: {mse_ga}")
        mse_results['GA'].append(mse_ga)

        # Only optimize eta and b (no omega optimization)
        print("Running AO algorithm optimizing only eta and b...")
        _, _, _, mse_list_no = ao_algorithm(
            h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_init,
            P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
            gamma_param=gamma_param, optimize_omega_func=no_optimize_omega,
            max_iter=max_iter, tol=tol
        )
        mse_no = mse_list_no[-1]
        print(f"Final MSE optimizing only eta and b: {mse_no}")
        mse_results['Only Optimize eta and b'].append(mse_no)

    return mse_results
# Main program execution
if __name__ == "__main__":
    # Hyperparameter settings
    K = 20  # Number of users
    N = 200  # Number of RIS elements (fixed in this script)
    max_iter = 50  # Maximum iterations
    tol = 1e-3  # Convergence threshold
    alpha_min = 0.2  # Minimum RIS reflection coefficient (Note: Might need adjustment for passive RIS)
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K) * 1  # Transmit power constraint for each user

    # List of noise power levels in dB (similar to chart range)
    sigma_a2_dB_list = [-120, -110, -100, -90, -80, -70, -60, -50, -40]

    # Run experiments and get MSE results
    mse_results = run_experiment(
        sigma_a2_dB_list=sigma_a2_dB_list, K=K, N=N, max_iter=max_iter, tol=tol,
        alpha_min=alpha_min, phi=phi, gamma_param=gamma_param, P_k=P_k
    )

    # Plot MSE vs. noise power curve using a log scale for MSE
    plt.figure(figsize=(12, 8))
    plt.semilogy(sigma_a2_dB_list, mse_results['GA'], 'r-o', label='GA Optimization', linewidth=2, markersize=8)
    plt.semilogy(sigma_a2_dB_list, mse_results['Only Optimize eta and b'], 'b-s', label='Only Optimize η and b', linewidth=2, markersize=8)
    plt.semilogy(sigma_a2_dB_list, mse_results['No Optimization at All'], 'k--', label='No Optimization at All', linewidth=2, markersize=8)

    plt.xlabel('Noise power $\sigma_a^2$ (dB)', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('MSE with Noise Power for Different Optimization Methods', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.savefig("mse_vs_noise_power.png") # Uncomment to save the figure
    plt.show()

    # Print final MSE results
    print("\nFinal MSE Results:")
    for method, mses in mse_results.items():
        print(f"{method} Optimization Method MSEs: {mses}")