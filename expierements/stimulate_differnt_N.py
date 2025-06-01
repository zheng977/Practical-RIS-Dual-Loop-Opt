# -*- coding: utf-8 -*-
"""
Main program to execute the Alternating Optimization (AO) algorithm and compare
the effectiveness of different optimization methods (e.g., SCA, GA, PSO, no optimization)
under varying numbers of RIS elements N.
"""

import numpy as np
import matplotlib.pyplot as plt
# from channel_simulation import generate_channels_new # Original, likely unused or needs adjustment
from ..ao_algorithm import ao_algorithm # Adjusted import
from ..optimizers import ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca # Adjusted import
from ..channel import channel_simulation # Adjusted import
from ..location import location # Adjusted import
import random
import torch
import os
# Set Chinese font to avoid Matplotlib font warnings (can be removed if not needed)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
os.environ['PYTHONHASHSEED'] = '2024'  # Control hash randomness

def calculate_mse(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param):
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

def run_experiment(N_list, K, sigma_a2, max_iter, tol, alpha_min, phi, gamma_param, P_k):
    mse_results = {
        'GA': [],
        'SCA': [],
        'Only Optimize eta and b': [],
        'No Optimization at All': []
    }

    for N_val in N_list: # Renamed N to N_val to avoid conflict with outer scope N if any
        print(f"\nRunning experiment with number of RIS elements N = {N_val}...")
        # Generate channels
        pL_UA, pL_UR, pL_RA = location(K)
        _, h, g = channel_simulation(pL_UA, pL_UR, pL_RA, K, N_val, 1) # Use N_val, Assuming M=1 for AP
        g = g.squeeze() # Ensure g is a vector

        print(f"Channel matrix h shape: {h.shape}")
        print(f"Channel vector g shape: {g.shape}")

        # Initialize variables
        eta_init = 1.0
        b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
        omega_init = np.random.uniform(low=-1*np.pi, high=1*np.pi, size=N_val) # Use N_val

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
            gamma_param=gamma_param, optimize_omega_func=no_optimize_omega, # Omega fixed after GA
            max_iter=max_iter, tol=tol
        )
        mse_ga = mse_list_ga[-1] if mse_list_ga else np.nan # Handle empty list
        print(f"Final MSE with GA optimization: {mse_ga}")
        mse_results['GA'].append(mse_ga)

        # SCA optimization
        print("Running AO algorithm with omega_n optimized by SCA...")
        _, _, _, mse_list_sca = ao_algorithm(
            h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_init,
            P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
            gamma_param=gamma_param, optimize_omega_func=optimize_omega_sca,
            max_iter=max_iter, tol=tol
        )
        mse_sca = mse_list_sca[-1] if mse_list_sca else np.nan # Handle empty list
        print(f"Final MSE with SCA optimization: {mse_sca}")
        mse_results['SCA'].append(mse_sca)

        # No omega optimization
        print("Running AO algorithm without optimizing omega_n...")
        _, _, _, mse_list_no = ao_algorithm(
            h=h, g=g, eta_init=eta_init, b_init=b_init, omega_init=omega_init,
            P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
            gamma_param=gamma_param, optimize_omega_func=no_optimize_omega,
            max_iter=max_iter, tol=tol
        )
        mse_no = mse_list_no[-1] if mse_list_no else np.nan # Handle empty list
        print(f"Final MSE without omega optimization: {mse_no}")
        mse_results['Only Optimize eta and b'].append(mse_no)

    return mse_results

# Main program execution
if __name__ == "__main__":
    # Hyperparameter settings
    K_users = 20  # Number of users, renamed from K to avoid conflict with P_k
    sigma_a2_noise = 1e-11  # AP noise variance
    max_iterations = 10  # Maximum iterations
    tolerance = 1e-3  # Convergence threshold
    alpha_min_coeff = 0.2  # Minimum RIS reflection coefficient
    phi_shift = 0.0  # Phase shift parameter
    gamma_parameter = 3.0  # Parameter controlling the steepness of the function curve
    P_k_power = np.ones(K_users) * 1  # Transmit power constraint for each user

    # List of RIS element counts for experiments
    N_values = [100, 200, 300, 400, 500]

    # Run experiments and get MSE results
    mse_data = run_experiment(
        N_list=N_values, K=K_users, sigma_a2=sigma_a2_noise, max_iter=max_iterations, tol=tolerance,
        alpha_min=alpha_min_coeff, phi=phi_shift, gamma_param=gamma_parameter, P_k=P_k_power
    )

    # Plot MSE vs. N curve using a log scale for MSE
    plt.figure(figsize=(12, 8))
    if mse_data['GA']: plt.semilogy(N_values, mse_data['GA'], 'r-o', label='GA Optimization', linewidth=2, markersize=8)
    if mse_data['SCA']: plt.semilogy(N_values, mse_data['SCA'], 'b-s', label='SCA Optimization', linewidth=2, markersize=8)
    if mse_data['Only Optimize eta and b']: plt.semilogy(N_values, mse_data['Only Optimize eta and b'], 'k--', label='Only Optimize η and b', linewidth=2, markersize=8)
    if mse_data['No Optimization at All']: plt.semilogy(N_values, mse_data['No Optimization at All'], 'm-.', label='No Optimization at All', linewidth=2, markersize=8)
    plt.xlabel('Number of RIS Elements N', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('MSE vs. Number of RIS Elements N for Different Optimization Methods', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.savefig("mse_vs_N_elements.png") # Uncomment to save the figure
    plt.show()

    # Print final MSE results
    print("\nFinal MSE Results:")
    for method, mses in mse_data.items():
        print(f"{method} Optimization Method MSEs: {mses}")