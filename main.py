import numpy as np
import matplotlib.pyplot as plt
from ao_algorithm import ao_algorithm
from optimizers import  ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca
from channel import channel_simulation
import torch
from location import location
"""
Main program to execute the Alternating Optimization (AO) algorithm and compare
the performance of different optimization methods (e.g., SCA, GA, PSO, and no optimization).
"""
# ====================== Hyperparameter Settings ======================
K = 20 # Number of users
N = 200 # Number of RIS elements
sigma_a2 = 1e-10  # AP noise variance
max_iter = 50  # Maximum iterations
tol = 1e-3  # Convergence threshold
alpha_min = 0.2  # Minimum RIS reflection coefficient
phi = 0.0  # Phase shift parameter
gamma_param = 3.0  # Parameter controlling the steepness of the function curve
P_k = np.ones(K)*1  # Transmit power constraint for each user, assuming all are 1

# GA parameters
population_size = 80  # Population size
num_generations = 100  # Number of generations
crossover_rate = 0.8  # Crossover rate
mutation_rate = 0.3  # Mutation rate

# PSO parameters
num_particles = 200  # Number of particles
max_iterations_pso = 200  # PSO maximum iterations
w = 0.7  # Inertia weight
c1 = 3.0  # Individual acceleration factor
c2 = 3.0  # Social acceleration factor

# Set Chinese font to avoid Matplotlib font warnings (can be removed if not needed)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
np.random.seed(2024)
torch.manual_seed(2024)
pL_UA, pL_UR, pL_RA = location(K)
_, h, g = channel_simulation(pL_UA, pL_UR, pL_RA, K, N, 1) # Assuming M=1 for AP antennas
g = g.squeeze() # Ensure g is a vector
print(f"Channel matrix h shape: {h.shape}")
print(f"Channel vector g shape: {g.shape}")

# Initialize variables
eta_init = 1.0
b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
omega_init = np.random.uniform(low=-1*np.pi, high=1*np.pi, size=N)

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

print(f"Final MSE with GA optimization: {mse_list_ga[-1] if mse_list_ga else 'N/A'}")

print("Running AO algorithm with omega_n optimized by SCA...")
eta_opt_sca, b_opt_sca, omega_opt_sca, mse_list_sca = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=optimize_omega_sca, max_iter=max_iter, tol=tol)

print(f"\nFinal MSE with SCA: {mse_list_sca[-1] if mse_list_sca else 'N/A'}")

# PSO optimization
print("Running AO algorithm with omega_n optimized by PSO...")
omega_opt_pso = pso_optimize_omega(h=h, g=g, eta=eta_init, b=b_init,
    omega_old=omega_init, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    sigma_a2=sigma_a2, num_particles=num_particles, max_iterations=max_iterations_pso,
    w=w, c1=c1, c2=c2)

print("Performing AO optimization using the best phase found by PSO...")
mse_list_pso = [] # Initialize mse_list_pso for the case where it might not be populated
eta_opt_pso, b_opt_pso, omega_opt_pso, mse_list_pso = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_opt_pso, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=no_optimize_omega, max_iter=max_iter, tol=tol)

print(f"Final MSE with PSO optimization: {mse_list_pso[-1] if mse_list_pso else 'N/A'}")

print("Running AO algorithm without optimizing omega_n...")
eta_opt_no, b_opt_no, omega_opt_no, mse_list_no = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=no_optimize_omega, max_iter=max_iter, tol=tol)

print(f"Final MSE without omega optimization: {mse_list_no[-1] if mse_list_no else 'N/A'}")

# Plot MSE vs. iteration number using a logarithmic scale
plt.figure(figsize=(12, 8))
if mse_list_ga: plt.semilogy(range(1, len(mse_list_ga) + 1), mse_list_ga, 'r-o', label='GA Optimization')
if mse_list_sca: plt.semilogy(range(1, len(mse_list_sca) + 1), mse_list_sca, 'b-s', label='SCA Optimization')
if mse_list_pso: plt.semilogy(range(1, len(mse_list_pso) + 1), mse_list_pso, 'g-^', label='PSO Optimization')
if mse_list_no: plt.semilogy(range(1, len(mse_list_no) + 1), mse_list_no, 'k--', label='No Omega Optimization')
plt.xlabel('Iteration Number', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('MSE vs. Iteration Number for Different Optimizers', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
# plt.savefig("main_mse_iterations_plot.png") # Uncomment to save the figure
plt.show()

# Output optimization results (optional, uncomment if needed)
# print("\nResults using GA optimization:")
# print("Optimized eta value:", eta_opt_ga)
# print("Optimized b_k values:", b_opt_ga)
# print("Optimized omega_n values:", omega_opt_ga)
#
# print("\nResults using SCA optimization:")
# print("Optimized eta value:", eta_opt_sca)
# print("Optimized b_k values:", b_opt_sca)
# print("Optimized omega_n values:", omega_opt_sca)
#
# print("\nResults using PSO optimization:")
# print("Optimized eta value:", eta_opt_pso)
# print("Optimized b_k values:", b_opt_pso)
# print("Optimized omega_n values:", omega_opt_pso)
#
# print("\nResults without omega optimization:")
# print("Optimized eta value:", eta_opt_no)
# print("Optimized b_k values:", b_opt_no)
# print("Optimized omega_n values:", omega_opt_no)