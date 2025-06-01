
"""
Implementation of the Alternating Optimization (AO) algorithm.
Used to optimize eta, b, and omega to minimize Mean Squared Error (MSE).
Supports using different optimizer functions to optimize omega_n.
"""
import numpy as np
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

def ao_algorithm(h, g, eta_init, b_init, omega_init, P_k, sigma_a2, alpha_min, phi, gamma_param,
                optimize_omega_func, max_iter=100, tol=1e-4, **kwargs):
    """
    Executes the Alternating Optimization (AO) algorithm to optimize eta, b, and omega_n,
    aiming to minimize MSE.

    Parameters:
    - h (numpy.ndarray): Channel matrix from users to RIS, shape (N, K)
    - g (numpy.ndarray): Channel vector from RIS to AP, shape (N,)
    - eta_init (float): Initial eta value
    - b_init (numpy.ndarray): Initial b vector, shape (K,)
    - omega_init (numpy.ndarray): Initial omega vector, shape (N,)
    - P_k (numpy.ndarray): Transmit power constraint for each user, shape (K,)
    - sigma_a2 (float): Noise variance at the AP
    - alpha_min (float): Minimum reflection coefficient of the RIS
    - phi (float): Phase shift parameter
    - gamma_param (float): Parameter controlling the steepness of the function curve
    - optimize_omega_func (function): Function to optimize omega_n
    - max_iter (int): Maximum number of iterations
    - tol (float): Convergence threshold
    - **kwargs: Additional arguments to pass to the optimizer function

    Returns:
    - eta (float): Optimized eta value
    - b (numpy.ndarray): Optimized b vector, shape (K,)
    - omega (numpy.ndarray): Optimized omega vector, shape (N,)
    - mse_list (list): MSE values for each iteration
    """
    K = h.shape[1]  # Number of users
    N = h.shape[0]  # Number of RIS elements

    eta = eta_init
    b = b_init.copy()
    omega = omega_init.copy()

    mse_list = []  # Store MSE value for each iteration

    for iter_num in range(max_iter):
        # Step 1: Update eta
        alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min
        Phi = np.diag(alpha * np.exp(1j * omega))  # RIS reflection matrix
        h_e = g.conj().T @ Phi @ h  # Equivalent channel, shape (K,)

        numerator = np.sum(np.real(h_e * b))
        denominator = np.sum(np.abs(h_e * b) ** 2) + sigma_a2
        eta_new = (1 / K) * numerator / denominator

        # Step 2: Update b_k
        b_new = np.zeros_like(b, dtype=complex)
        for k in range(K):
            h_e_k = h_e[k]
            if np.abs(h_e_k) == 0:
                # Avoid division by zero
                b_unconstrained = 0
            else:
                b_unconstrained = h_e_k.conj() / (eta_new * K * np.abs(h_e_k) ** 2)
            if np.abs(b_unconstrained) ** 2 <= P_k[k]:
                b_new[k] = b_unconstrained
            else:
                b_new[k] = np.sqrt(P_k[k]) * np.exp(1j * np.angle(h_e_k.conj()))

        # Step 3: Optimize omega_n
        # Call the passed optimizer function, ensure parameter names match
        omega_new = optimize_omega_func(
            h=h,
            g=g,
            eta=eta_new,
            b=b_new,
            omega_old=omega,
            alpha_min=alpha_min,
            phi=phi,
            gamma_param=gamma_param,
            sigma_a2=sigma_a2,
            **kwargs
        )

        mse = calculate_mse(
            h=h,  # Channel matrix
            g=g,  # Channel vector
            eta=eta_new,  # Optimized eta
            b=b_new,  # Optimized b
            omega=omega_new,  # Optimized omega
            sigma_a2=sigma_a2,  # Noise variance
            alpha_min=alpha_min,  # Minimum reflection coefficient
            phi=phi,  # Phase shift
            gamma_param=gamma_param  # gamma parameter
        )
        # Check for convergence
        if iter_num > 0 and ((np.abs(mse- mse_list[-1])/mse_list[-1]) < tol ):
            break

        mse_list.append(mse)


        # Update variables for the next iteration
        eta = eta_new
        b = b_new
        omega = omega_new

    # Return optimized variables and MSE list
    return eta, b, omega, mse_list
