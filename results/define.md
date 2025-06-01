The parameters used to generate each figure are as follows:

Iteration.png:
![Iteration.png](Iteration.png)

    K = 20 # Number of users
    N = 200 # Number of RIS elements
    sigma_a2 = 1e-12  # AP noise variance, set to 1e-12
    max_iter = 30  # Maximum iterations
    tol = 1e-11 # Convergence threshold
    alpha_min = 0.2  # Minimum RIS reflection coefficient
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K)*1  # Transmit power constraint for each user, assuming all are 1

different_dB_s_MSE.png (Note: consider renaming the file to not have spaces for better compatibility)
![different dB 's MSE.png](different%20dB%20%27s%20MSE.png)
    
    K = 20  # Number of users
    N = 200  # Number of RIS elements (fixed)
    max_iter = 50  # Maximum iterations
    tol = 1e-3  # Convergence threshold
    alpha_min = 0.2  # Minimum RIS reflection coefficient (Note: may need adjustment to 0.2 for passive RIS)
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K) * 1  # Transmit power constraint for each user

different_alpha_s_MSE.png (Note: consider renaming the file to not have spaces for better compatibility)
![different alpha 's MSE.png](different%20alpha%20%27s%20MSE.png)
    
    K = 20  # Number of users
    N = 200  # Number of RIS elements
    sigma_a2 = 1e-11  # AP noise variance
    max_iter = 20  # Maximum iterations
    tol = 0 # Convergence threshold
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K) * 1  # Transmit power constraint for each user, assuming all are 1


Different_RIS_N.png
![Diferent_RIS_N.png](Diferent_RIS_N.png)
    
    K = 20  # Number of users
    sigma_a2 = 1e-11  # AP noise variance
    max_iter = 15  # Maximum iterations
    tol = 1e-3  # Convergence threshold
    alpha_min = 0.2  # Minimum RIS reflection coefficient
    phi = 0.0  # Phase shift parameter
    gamma_param = 3.0  # Parameter controlling the steepness of the function curve
    P_k = np.ones(K) * 1  # Transmit power constraint for each user



