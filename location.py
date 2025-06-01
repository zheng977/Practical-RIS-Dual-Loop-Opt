import numpy as np


def path_loss(T0, d, d0, a):
    """
    Calculates path loss.
    T0: Reference path loss
    d: Actual distance
    d0: Reference distance
    a: Path loss exponent
    """
    return T0 * (d / d0) ** (-a)


def location(K):
    """
    Generates locations for users, RIS, and AP, along with path losses.
    K: Number of users
    Returns: pL_UA, pL_UR, pL_RA (Path losses for User-AP, User-RIS, RIS-AP)
    """
    # Fixed AP and RIS locations
    location_AP = np.array([-50, 0, 12])
    location_RIS = np.array([0, 0, 12])

    # Path loss parameters
    T0 = 1e-3
    a_UA = 3.6  # User-AP link
    a_UR = 2.8  # User-RIS link
    a_RA = 2.2  # RIS-AP link

    # Calculate RIS-AP path loss
    dis_RA = np.linalg.norm(location_AP - location_RIS)
    pL_RA = path_loss(T0, dis_RA, 1, a_RA)

    # Initialize path losses and user locations
    pL_UA = np.zeros(K)
    pL_UR = np.zeros(K)
    location_user = np.zeros((K, 3))

    # Randomly generate user locations
    for k in range(K):
        x = np.random.uniform(0, 20)  # Range [0, 20]
        y = np.random.uniform(-10, 10)  # Range [-10, 10]
        z = 0
        location_user[k, :] = [x, y, z]

    # Calculate User-AP and User-RIS path losses
    for k in range(K):
        dis_UA = np.linalg.norm(location_AP - location_user[k, :])
        pL_UA[k] = path_loss(T0, dis_UA, 1, a_UA)

        dis_UR = np.linalg.norm(location_RIS - location_user[k, :])
        pL_UR[k] = path_loss(T0, dis_UR, 1, a_UR)

    return pL_UA, pL_UR, pL_RA