# -*- coding: utf-8 -*-
"""
Main program to execute the Alternating Optimization (AO) algorithm and compare
the performance of different optimization methods (e.g., SCA, GA, PSO, neural network, and no optimization).
Exports MSE values for all iterations to an Excel file for plotting in Origin.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import squeeze
# from channel_simulation import generate_channels_new # Original, likely needs adjustment or removal
from ..ao_algorithm import ao_algorithm # Adjusted import
from ..optimizers import ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca, optimize_omega_sca_2nd_bound, optimize_omega_sca2 # Adjusted import
from ..channel import channel_simulation # Adjusted import
# from channel_simulation import generate_channels_new # Duplicate, remove if not used or from a different path
import torch
from ..location import location # Adjusted import
import pandas as pd
plt.rcParams['font.family'] = 'STIXGeneral'  # Or other font supporting math symbols, e.g., 'DejaVu Sans'
import cvxpy

def calculate_mse(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param):
    """
    Directly calculates the MSE value without printing intermediate information.

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

# ====================== Hyperparameter Settings ======================
K = 20  # Number of users
N = 200  # Number of RIS elements
sigma_a2 = 1e-12  # AP noise variance, set to 1e-12
max_iter = 30  # Maximum iterations
tol = 1e-11  # Convergence threshold
alpha_min = 0.2  # Minimum RIS reflection coefficient
phi = 0.0  # Phase shift parameter
gamma_param = 3.0  # Parameter controlling the steepness of the function curve
P_k = np.ones(K) * 1  # Transmit power constraint for each user, assuming all are 1

# GA parameters
population_size = 80  # Population size
num_generations = 100  # Number of generations
crossover_rate = 0.8  # Crossover rate
mutation_rate = 0.3  # Mutation rate

# PSO parameters (defined but pso_optimize_omega is not called in this script version)
# num_particles = 100  # Number of particles
# max_iterations_pso = 500  # PSO maximum iterations
# w = 0.7  # Inertia weight
# c1 = 2.0  # Individual acceleration factor
# c2 = 2.0  # Social acceleration factor

# Set random seed for reproducibility
np.random.seed(2024)
torch.manual_seed(2024)

pL_UA, pL_UR, pL_RA = location(K)
# location_AP = np.array([-50, 0, 12]) # Defined in location.py
# location_RIS = np.array([0, 0, 12]) # Defined in location.py
_, h, g = channel_simulation(pL_UA, pL_UR, pL_RA, K, N, 1) # Assuming M=1 for AP antennas
g = g.squeeze() # Ensure g is a vector

# Initialize variables
eta_init = 1.0
b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
omega_init = np.random.uniform(low=-0 * np.pi, high=0 * np.pi, size=N) # Initial omega is all zeros

# Calculate baseline MSE for no optimization
mse_base = calculate_mse(
    h=h, g=g, eta=eta_init, b=b_init, omega=omega_init,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param
)
print(f"Baseline MSE with no optimization: {mse_base}")

# To record the baseline for no optimization in each iteration, create an array of length max_iter
mse_base_list = np.ones(max_iter) * mse_base

# GA optimization
print("Running AO algorithm with omega_n optimized by GA...")
omega_opt_ga = ga_optimize_omega(h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
                            alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
                            sigma_a2=sigma_a2, population_size=population_size,
                            num_generations=num_generations, crossover_rate=crossover_rate,
                            mutation_rate=mutation_rate)

print("Performing AO optimization using the best phase found by GA...")
eta_opt_ga, b_opt_ga, omega_opt_ga, mse_list_ga = ao_algorithm(h=h, g=g, eta_init=eta_init,
    b_init=b_init, omega_init=omega_opt_ga, P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min,
    phi=phi, gamma_param=gamma_param, optimize_omega_func=no_optimize_omega, # Omega is fixed after GA
    max_iter=max_iter, tol=tol)

print(f"Final MSE with GA optimization: {mse_list_ga[-1]}")

print("Running AO algorithm with omega_n optimized by SCA...")
eta_opt_sca, b_opt_sca, omega_opt_sca, mse_list_sca = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=optimize_omega_sca, max_iter=max_iter, tol=tol)

print(f"Final MSE with SCA optimization: {mse_list_sca[-1]}")

print("Running AO algorithm without optimizing omega_n...")
eta_opt_no, b_opt_no, omega_opt_no, mse_list_no = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=no_optimize_omega, max_iter=max_iter, tol=tol)

print(f"Final MSE without omega optimization: {mse_list_no[-1]}")

# Find the maximum length of MSE lists to align all data
max_length = max(len(mse_list_ga), len(mse_list_sca), len(mse_list_no), len(mse_base_list))
if max_length == 0: max_length = 1 # Ensure max_length is at least 1

# Extend all MSE lists to the same length (pad with the last value)
def extend_list(lst, length):
    if not lst: # Handle empty list case
        return [np.nan] * length # Use NaN for padding if list is empty
    if len(lst) < length:
        return list(lst) + [lst[-1]] * (length - len(lst))
    return lst

mse_list_ga_extended = extend_list(mse_list_ga, max_length)
mse_list_sca_extended = extend_list(mse_list_sca, max_length)
mse_list_no_extended = extend_list(mse_list_no, max_length)
mse_base_list_extended = extend_list(mse_base_list, max_length)

# Create a DataFrame to store all MSE values
mse_df = pd.DataFrame({
    'Iteration': range(1, max_length + 1),
    'GA Optimization': mse_list_ga_extended,
    'SCA Optimization': mse_list_sca_extended,
    'Only Optimize eta and b': mse_list_no_extended,
    'No Optimization at All': mse_base_list_extended
})

# Export to Excel file
excel_filename = 'mse_iterations_data.xlsx' # Note: File saved in current directory
mse_df.to_excel(excel_filename, index=False)
print(f"All MSE iteration data exported to {excel_filename}")

# Print all iteration MSE values for each method
print("\nGA Optimization MSE values for all iterations:")
for i, mse in enumerate(mse_list_ga):
    print(f"Iteration {i+1}: {mse}")

print("\nSCA Optimization MSE values for all iterations:")
for i, mse in enumerate(mse_list_sca):
    print(f"Iteration {i+1}: {mse}")

print("\nOnly Optimize eta and b MSE values for all iterations:")
for i, mse in enumerate(mse_list_no):
    print(f"Iteration {i+1}: {mse}")

# Plot MSE vs. iteration number using a logarithmic scale
plt.figure(figsize=(12, 8))
if mse_list_ga: plt.semilogy(range(1, len(mse_list_ga) + 1), mse_list_ga, 'r-o', label='GA Optimization')
if mse_list_sca: plt.semilogy(range(1, len(mse_list_sca) + 1), mse_list_sca, 'b-s', label='SCA Optimization')
if mse_list_no: plt.semilogy(range(1, len(mse_list_no) + 1), mse_list_no, 'k--', label='Only Optimize eta and b')
if mse_base_list.size > 0 and not np.all(np.isnan(mse_base_list)): # Ensure base list is not all NaN
    plt.semilogy(range(1, len(mse_base_list) + 1), mse_base_list, 'm-.', label='No Optimization at All')
plt.xlabel('Iteration Number', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('MSE vs. Iteration Number', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('mse_iterations_plot.png', dpi=300) # Note: Image saved in current directory
plt.show()