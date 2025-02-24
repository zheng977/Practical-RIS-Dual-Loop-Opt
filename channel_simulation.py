import numpy as np

def generate_channels_new(N, K, pl_RA, pl_UA, location_user, location_AP, location_RIS, gamma_db=3):
    """
    基于给定路径损耗和位置信息生成信道（支持单天线）。
    """

    def calculate_angle(pos1, pos2):
        delta = pos2 - pos1
        return np.arctan2(delta[1], delta[0])

    gamma = 10 ** (gamma_db / 10)

    # 生成RIS到AP的信道（单天线方向矢量为1）
    g_nlos = np.sqrt(0.5) * (np.random.randn(N) + 1j * np.random.randn(N))
    g_los = np.sqrt(pl_RA) * 1  # 方向矢量退化为1
    g_channel = np.sqrt(gamma / (1 + gamma)) * g_los + np.sqrt(1 / (1 + gamma)) * g_nlos

    # 生成用户到RIS的信道（单天线方向矢量为1）
    h = np.zeros((N, K), dtype=complex)
    for k in range(K):
        h_nlos = np.sqrt(0.5) * (np.random.randn(N) + 1j * np.random.randn(N))
        h_los = np.sqrt(pl_UA[k]) * 1  # 方向矢量退化为1
        h[:, k] = np.sqrt(gamma / (1 + gamma)) * h_los + np.sqrt(1 / (1 + gamma)) * h_nlos

    # 归一化
    g_channel = g_channel / np.sqrt(np.mean(np.abs(g_channel) ** 2))
    h = h / np.sqrt(np.mean(np.abs(h) ** 2))

    return h, g_channel.reshape(-1, 1)