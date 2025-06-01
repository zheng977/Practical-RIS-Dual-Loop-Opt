"""
Main program to execute the Alternating Optimization (AO) algorithm,
comparing the iteration process of various optimization methods under different minimum reflection coefficients (alpha_min).
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import squeeze
# from channel_simulation import generate_channels_new # Original, likely needs adjustment or removal if not used
from ..ao_algorithm import ao_algorithm # Adjusted import
from ..optimizers import ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca # Adjusted import
from ..channel import channel_simulation # Adjusted import
# from channel_simulation import generate_channels_new # Duplicate, remove if generate_channels_new is not from a different path or used
import torch
from ..location import location # Adjusted import
import pandas as pd
import cvxpy

# Set Chinese font to avoid Matplotlib font warnings (can be removed if not needed)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
p.random.seed(2024)
torch.manual_seed(2024)


def run_experiment_for_alpha_min(alpha_min):
    """
    Runs the experiment for a specific alpha_min value and returns the iterative MSE values for all algorithms.

    Parameters:
    - alpha_min: Minimum reflection coefficient of the RIS

    Returns:
    - Iterative MSE lists for different optimization algorithms
    """
    print(f"\n=== Running experiment for alpha_min = {alpha_min} ===")

    # ====================== Hyperparameter Settings ======================
    K = 20  # Number of users
    N = 200  # Number of RIS elements
    sigma_a2 = 1e-11  # AP noise variance
    max_iter = 20  # Maximum iterations
    tol = 0 # Convergence threshold
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K) * 1  # Transmit power constraint for each user, assuming all are 1

    # GA parameters
    population_size = 80  # Population size
    num_generations = 100  # Number of generations
    crossover_rate = 0.8  # Crossover rate
    mutation_rate = 0.3  # Mutation rate

    # PSO parameters (currently unused in this script but defined)
    # num_particles = 200  # Number of particles
    # max_iterations_pso = 200  # PSO maximum iterations
    # w = 0.7  # Inertia weight
    # c1 = 3.0  # Individual acceleration factor
    # c2 = 3.0  # Social acceleration factor

    # Generate channels (same channels used for all experiments)
    pL_UA, pL_UR, pL_RA = location(K)
    _, h, g = channel_simulation(pL_UA, pL_UR, pL_RA, K, N, 1) # Assuming M=1 for AP antennas
    g = g.squeeze() # Ensure g is a vector

    # Initialize variables
    eta_init = 1.0
    b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
    omega_init = np.random.uniform(low=-1 * np.pi, high=1 * np.pi, size=N)

    # GA optimization
    print("Running AO algorithm with omega_n optimized by GA...")
    omega_opt_ga = ga_optimize_omega(h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
                                  alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
                                  sigma_a2=sigma_a2, population_size=population_size,
                                  num_generations=num_generations, crossover_rate=crossover_rate,
                                  mutation_rate=mutation_rate)

    print("Performing AO optimization using the best phase found by GA...")
    eta_opt_ga, b_opt_ga, omega_opt_ga, mse_list_ga = ao_algorithm(h=h, g=g, eta_init=eta_init,
                                                                   b_init=b_init, omega_init=omega_opt_ga, P_k=P_k,
                                                                   sigma_a2=sigma_a2, alpha_min=alpha_min,
                                                                   phi=phi, gamma_param=gamma_param,
                                                                   optimize_omega_func=no_optimize_omega, # Omega is fixed after GA
                                                                   max_iter=max_iter, tol=tol)

    print(f"Final MSE with GA optimization: {mse_list_ga[-1]}")

    # SCA optimization
    print("Running AO algorithm with omega_n optimized by SCA...")
    eta_opt_sca, b_opt_sca, omega_opt_sca, mse_list_sca = ao_algorithm(h=h, g=g,
                                                                       eta_init=eta_init, b_init=b_init,
                                                                       omega_init=omega_init, P_k=P_k,
                                                                       sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
                                                                       gamma_param=gamma_param,
                                                                       optimize_omega_func=optimize_omega_sca,
                                                                       max_iter=max_iter, tol=tol)

    print(f"Final MSE with SCA optimization: {mse_list_sca[-1]}")

    # No omega optimization
    print("Running AO algorithm without optimizing omega_n...")
    eta_opt_no, b_opt_no, omega_opt_no, mse_list_no = ao_algorithm(h=h, g=g,
                                                                   eta_init=eta_init, b_init=b_init,
                                                                   omega_init=omega_init, P_k=P_k,
                                                                   sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
                                                                   gamma_param=gamma_param,
                                                                   optimize_omega_func=no_optimize_omega,
                                                                   max_iter=max_iter, tol=tol)

    print(f"Final MSE without omega optimization: {mse_list_no[-1]}")

    # Print MSE values for all iterations
    print(f"\n=== GA Optimization MSE Iteration Values for alpha_min = {alpha_min} ===")
    for i, mse in enumerate(mse_list_ga):
        print(f"Iteration {i + 1}: {mse}")

    print(f"\n=== SCA Optimization MSE Iteration Values for alpha_min = {alpha_min} ===")
    for i, mse in enumerate(mse_list_sca):
        print(f"Iteration {i + 1}: {mse}")

    print(f"\n=== No Omega Optimization MSE Iteration Values for alpha_min = {alpha_min} ===")
    for i, mse in enumerate(mse_list_no):
        print(f"Iteration {i + 1}: {mse}")

    return mse_list_ga, mse_list_sca, mse_list_no


# Run experiment for alpha_min = 0.2
mse_list_ga_0_2, mse_list_sca_0_2, mse_list_no_0_2 = run_experiment_for_alpha_min(0.2)

# Run experiment for alpha_min = 1.0
mse_list_ga_1_0, mse_list_sca_1_0, mse_list_no_1_0 = run_experiment_for_alpha_min(1.0)

# Save all iterative MSEs to an Excel file
# Find the maximum length of MSE lists
max_length = max(len(mse_list_ga_0_2), len(mse_list_sca_0_2), len(mse_list_no_0_2),
                 len(mse_list_ga_1_0), len(mse_list_sca_1_0), len(mse_list_no_1_0))


# Extend all lists to the same length (pad with the last value)
def extend_list(lst, length):
    if not lst: # Handle empty list case
        return [np.nan] * length # Or some other placeholder like 0 or None
    if len(lst) < length:
        return list(lst) + [lst[-1]] * (length - len(lst))
    return lst


mse_ga_0_2_ext = extend_list(mse_list_ga_0_2, max_length)
mse_sca_0_2_ext = extend_list(mse_list_sca_0_2, max_length)
mse_no_0_2_ext = extend_list(mse_list_no_0_2, max_length)
mse_ga_1_0_ext = extend_list(mse_list_ga_1_0, max_length)
mse_sca_1_0_ext = extend_list(mse_list_sca_1_0, max_length)
mse_no_1_0_ext = extend_list(mse_list_no_1_0, max_length)

# Create DataFrame
mse_df = pd.DataFrame({
    'Iteration': range(1, max_length + 1),
    'GA (alpha_min=0.2)': mse_ga_0_2_ext,
    'SCA (alpha_min=0.2)': mse_sca_0_2_ext,
    'No Omega Opt (alpha_min=0.2)': mse_no_0_2_ext,
    'GA (alpha_min=1.0)': mse_ga_1_0_ext,
    'SCA (alpha_min=1.0)': mse_sca_1_0_ext,
    'No Omega Opt (alpha_min=1.0)': mse_no_1_0_ext
})

# Export to CSV file
csv_filename = 'mse_iterations_by_alpha_min.csv' # Note: File saved in current directory
mse_df.to_csv(csv_filename, index=False)
print(f"\nAll MSE iteration data exported to {csv_filename}")

# Plot comparison graph
plt.figure(figsize=(12, 8))

# Curves for alpha_min = 0.2
plt.semilogy(range(1, len(mse_list_ga_0_2) + 1), mse_list_ga_0_2, 'r-o', label='GA (α_min=0.2)', linewidth=2,
             markersize=8, markevery=1)
plt.semilogy(range(1, len(mse_list_sca_0_2) + 1), mse_list_sca_0_2, 'b-s', label='SCA (α_min=0.2)', linewidth=2,
             markersize=8, markevery=1)
plt.semilogy(range(1, len(mse_list_no_0_2) + 1), mse_list_no_0_2, 'g--', label='Only Opt. η,b (α_min=0.2)', linewidth=2,
             markevery=1)

# Curves for alpha_min = 1.0
plt.semilogy(range(1, len(mse_list_ga_1_0) + 1), mse_list_ga_1_0, 'r-^', label='GA (α_min=1.0)', linewidth=2,
             markersize=8, markevery=1)
plt.semilogy(range(1, len(mse_list_sca_1_0) + 1), mse_list_sca_1_0, 'b-d', label='SCA (α_min=1.0)', linewidth=2,
             markersize=8, markevery=1)
plt.semilogy(range(1, len(mse_list_no_1_0) + 1), mse_list_no_1_0, 'g-.', label='Only Opt. η,b (α_min=1.0)', linewidth=2,
             markevery=1)

plt.xlabel('Iteration Number', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('MSE vs. Iteration Number for Different α_min Values', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('mse_iterations_comparison_alpha_min.png', dpi=300) # Note: Image saved in current directory
plt.show()