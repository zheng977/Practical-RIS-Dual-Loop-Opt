import numpy as np

def rician_channel(sz1, sz2, beta):
    """
    Generates a Rician channel.
    sz1, sz2: Channel matrix dimensions
    beta: Rician factor
    """
    # LoS component
    h_LoS = np.ones((sz1, sz2))

    # NLoS component (Rayleigh fading)
    h_NLoS = np.sqrt(0.5) * (np.random.randn(sz1, sz2) + 1j * np.random.randn(sz1, sz2))

    # Combined channel
    h = np.sqrt(beta / (1 + beta)) * h_LoS + np.sqrt(1 / (1 + beta)) * h_NLoS
    return h

def channel_simulation(pL_UA, pL_UR, pL_RA, node_num, ris_ant, server_ant):
    """
    Channel simulation function.
    Parameters:
    pL_UA: User-AP path loss array
    pL_UR: User-RIS path loss array
    pL_RA: RIS-AP path loss
    node_num: Number of users (K)
    ris_ant: Number of RIS antennas (N)
    server_ant: Number of AP antennas (M)
    Returns:
    hd: Direct channel (User-AP)
    hr: User-RIS channel
    G: RIS-AP channel
    """
    # Set number of users, RIS antennas, and AP antennas
    K = node_num
    N = ris_ant
    M = server_ant

    # Rician factor settings
    beta_UA = 0  # User-AP link
    beta_UR = 0  # User-RIS link
    beta_RA = 10 ** (3 / 10)  # RIS-AP link

    # Generate three types of channels
    # 1. RIS-AP channel
    G = np.sqrt(pL_RA) * rician_channel(M, N, beta_RA)

    # 2. User-AP direct channel
    hd = rician_channel(M, K, beta_UA)
    for k in range(K):
        hd[:, k] = np.sqrt(pL_UA[k]) * hd[:, k]

    # 3. User-RIS channel
    hr = rician_channel(N, K, beta_UR)
    for k in range(K):
        hr[:, k] = np.sqrt(pL_UR[k]) * hr[:, k]

    return hd, hr, G