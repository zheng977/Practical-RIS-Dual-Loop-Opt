# optimizers.py
# -*- coding: utf-8 -*-
"""
Implementations of different optimizers for omega_n.
"""
import numpy as np
import cvxpy as cp
from deap import base, creator, tools, algorithms
import random
from pyswarms.single.global_best import GlobalBestPSO
from ao_algorithm import ao_algorithm

def optimize_omega_sca(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2, tau=1e-2):
    """
    Optimizes omega_n using the Sequential Convex Approximation (SCA) method,
    with a regularization term added to improve performance.

    Parameters:
    - h (numpy.ndarray): Channel matrix from users to RIS, shape (N, K)
    - g (numpy.ndarray): Channel vector from RIS to AP, shape (N,)
    - eta (float): Current eta value
    - b (numpy.ndarray): Current b vector, shape (K,)
    - omega_old (numpy.ndarray): Omega values from the previous iteration, shape (N,)
    - alpha_min (float): Minimum reflection coefficient of the RIS
    - phi (float): Phase shift parameter
    - gamma_param (float): Parameter controlling the steepness of the function curve
    - sigma_a2 (float): Noise variance at the AP
    - tau (float): Regularization parameter, default is 1e-3

    Returns:
    - omega_new (numpy.ndarray): Updated omega values, shape (N,)
    """
    K = h.shape[1]  # Number of users
    N = h.shape[0]  # Number of RIS elements

    # Calculate alpha_n and its derivative with respect to omega_n
    sin_term = (np.sin(omega_old - phi) + 1) / 2
    alpha_old = (1 - alpha_min) * sin_term ** gamma_param + alpha_min
    d_alpha = (1 - alpha_min) * gamma_param * sin_term ** (gamma_param - 1) * (np.cos(omega_old - phi) / 2)

    # Precompute constants
    e_jomega_old = np.exp(1j * omega_old)  # Shape (N,)
    c_nk = g.conj()[:, np.newaxis] * h      # Shape (N, K)

    # Initialize C_k and D_k
    C_k = np.zeros(K, dtype=complex)        # Shape (K,)
    D_k = np.zeros((K, N), dtype=complex)   # Shape (K, N)

    for k in range(K):
        A_n = alpha_old * e_jomega_old       # Shape (N,)
        B_n = (d_alpha + 1j * alpha_old) * e_jomega_old  # Shape (N,)

        C_k[k] = eta * b[k] * np.sum(c_nk[:, k] * A_n)  # Scalar
        D_k[k, :] = eta * b[k] * c_nk[:, k] * B_n      # Shape (N,)

    # Define optimization variable
    omega = cp.Variable(N)

    # Construct the objective function
    obj = 0
    for k in range(K):
        E_k = C_k[k] - 1 / K
        F_n = D_k[k, :]  # Shape (N,)

        # Calculate real and imaginary parts separately
        real_part = cp.real(E_k + F_n @ (omega - omega_old))
        imag_part = cp.imag(E_k + F_n @ (omega - omega_old))

        obj += real_part ** 2 + imag_part ** 2

    # Add regularization term
    obj += (tau / 2) * cp.sum_squares(omega - omega_old)

    # Define constraints
    constraints = [omega >= -np.pi, omega <= np.pi]

    # Formulate and solve the optimization problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Get the optimized omega values
    omega_new = omega.value

    return omega_new



def ga_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2,
                     population_size=80, num_generations=100, crossover_rate=0.8, mutation_rate=0.2):
    K = h.shape[1]
    N = h.shape[0]
    P_k = np.ones(K) * 1  # Transmit power constraint for each user, assuming all are 1

    # Create DEAP classes
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Fitness function 5, 7
    def fitness(individual):
        omega = np.array(individual)
        # Fix omega, perform optimization for eta and b
        _, _, _, mse_list = ao_algorithm(
            h=h,
            g=g,
            eta_init=eta,  # Use current eta as initial value
            b_init=b,      # Use current b as initial value
            omega_init=omega,  # Use candidate omega
            P_k=P_k,
            sigma_a2=sigma_a2,
            alpha_min=alpha_min,
            phi=phi,
            gamma_param=gamma_param,
            optimize_omega_func=no_optimize_omega,  # Use no_optimize_omega to fix omega
            max_iter=10,
            tol=1e-6
        )
        return (mse_list[-1],)  # Return the optimized MSE

    # Set up DEAP framework
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -np.pi, np.pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic algorithm operations
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=np.pi/10, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Apply genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate,
                                 ngen=num_generations, verbose=False)

    # Get the individual with the best fitness
    best_ind = tools.selBest(pop, 1)[0]
    omega_new = np.array(best_ind)

    return omega_new

def pso_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2,
                      num_particles=30, max_iterations=100, w=0.7, c1=1.5, c2=1.5):
    """
    Optimizes omega_n using Particle Swarm Optimization (PSO).
    """
    K = h.shape[1]
    N = h.shape[0]
    P_k = np.ones(K) * 1

    # Define new fitness function
    def fitness_function(x):
        # x.shape = (num_particles, N)
        mse = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            omega = x[i]
            # Perform eta and b optimization for each particle position
            _, _, _, mse_history = ao_algorithm(
                h=h,
                g=g,
                eta_init=eta,
                b_init=b,
                omega_init=omega,  # Use current particle position as omega
                P_k=P_k,
                sigma_a2=sigma_a2,
                alpha_min=alpha_min,
                phi=phi,
                gamma_param=gamma_param,
                optimize_omega_func=no_optimize_omega,  # Fix omega
                max_iter=10,  # Use fewer iterations to optimize eta and b
                tol=1e-6
            )
            mse[i] = mse_history[-1]  # Use the optimized MSE value
        return mse

    # PSO parameters
    options = {'c1': c1, 'c2': c2, 'w': w}

    # Define search space boundaries
    lower_bounds = -np.pi * np.ones(N)
    upper_bounds = np.pi * np.ones(N)
    bounds = (lower_bounds, upper_bounds)

    # Initialize PSO optimizer
    optimizer = GlobalBestPSO(
        n_particles=num_particles,
        dimensions=N,
        options=options,
        bounds=bounds
    )

    # Run PSO optimization
    cost, pos = optimizer.optimize(
        fitness_function,
        iters=max_iterations
    )

    # Return the best position (optimal phase configuration)
    omega_new = pos

    return omega_new

def no_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2):
    """
    Does not optimize omega_n, directly returns the old omega.

    Parameters:
    - h (numpy.ndarray): Channel matrix from users to RIS, shape (N, K)
    - g (numpy.ndarray): Channel vector from RIS to AP, shape (N,)
    - eta (float): Current eta value
    - b (numpy.ndarray): Current b vector, shape (K,)
    - omega_old (numpy.ndarray): Omega values from the previous iteration, shape (N,)
    - alpha_min (float): Minimum reflection coefficient of the RIS
    - phi (float): Phase shift parameter
    - gamma_param (float): Parameter controlling the steepness of the function curve
    - sigma_a2 (float): Noise variance at the AP

    Returns:
    - omega_new (numpy.ndarray): Unchanged omega values, shape (N,)
    """
    return omega_old.copy()

