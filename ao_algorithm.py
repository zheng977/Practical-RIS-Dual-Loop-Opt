# ao_algorithm.py
# -*- coding: utf-8 -*-
"""
交替优化（AO）算法实现，用于优化η、b和ω，以最小化均方误差（MSE）。支持使用不同的优化器函数来优化omega_n。
"""

import numpy as np
def calculate_mse(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param):
    """
    直接计算MSE值

    参数:
    - h: 用户到RIS的信道矩阵 (N, K)
    - g: RIS到AP的信道向量 (N,)
    - eta: 优化后的eta值
    - b: 优化后的b向量 (K,)
    - omega: 优化后的omega值 (N,)
    - sigma_a2: 噪声方差
    - alpha_min: RIS最小反射系数
    - phi: 相位偏移
    - gamma_param: gamma参数

    返回:
    - mse: 计算得到的MSE值
    """
    # 计算反射系数alpha
    alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min

    # 构建RIS反射矩阵Φ
    Phi = np.diag(alpha * np.exp(1j * omega))

    # 计算等效信道h_e
    h_e = g.conj().T @ Phi @ h

    # 计算MSE
    K = h.shape[1]
    mse = np.sum(np.abs(eta * h_e * b - 1 / K) ** 2) + eta ** 2 * sigma_a2

    return mse

def ao_algorithm(h, g, eta_init, b_init, omega_init, P_k, sigma_a2, alpha_min, phi, gamma_param,
                optimize_omega_func, max_iter=100, tol=1e-4, **kwargs):
    """
    执行交替优化（AO）算法，优化eta、b和omega_n，以最小化MSE。

    参数：
    - h (numpy.ndarray): 用户到RIS的信道矩阵，形状为 (N, K)
    - g (numpy.ndarray): RIS到AP的信道向量，形状为 (N,)
    - eta_init (float): 初始eta值
    - b_init (numpy.ndarray): 初始b向量，形状为 (K,)
    - omega_init (numpy.ndarray): 初始omega值，形状为 (N,)
    - P_k (numpy.ndarray): 每个用户的发射功率约束，形状为 (K,)
    - sigma_a2 (float): AP端噪声方差
    - alpha_min (float): RIS反射系数的最小值
    - phi (float): 相位偏移参数
    - gamma_param (float): 控制函数曲线陡度的参数
    - optimize_omega_func (function): 优化omega_n的函数
    - max_iter (int): 最大迭代次数
    - tol (float): 收敛阈值
    - **kwargs: 传递给优化器函数的额外参数

    返回：
    - eta (float): 优化后的eta值
    - b (numpy.ndarray): 优化后的b向量，形状为 (K,)
    - omega (numpy.ndarray): 优化后的omega值，形状为 (N,)
    - mse_list (list): 每次迭代的MSE值
    """
    K = h.shape[1]  # 用户数量
    N = h.shape[0]  # RIS元素数量

    eta = eta_init
    b = b_init.copy()
    omega = omega_init.copy()

    mse_list = []  # 存储每次迭代的MSE值

    for iter_num in range(max_iter):
        # 第一步：更新eta
        alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min
        Phi = np.diag(alpha * np.exp(1j * omega))  # RIS反射矩阵
        h_e = g.conj().T @ Phi @ h  # 等效信道，形状为 (K,)

        numerator = np.sum(np.real(h_e * b))
        denominator = np.sum(np.abs(h_e * b) ** 2) + sigma_a2
        eta_new = (1 / K) * numerator / denominator

        # 第二步：更新b_k
        b_new = np.zeros_like(b, dtype=complex)
        for k in range(K):
            h_e_k = h_e[k]
            if np.abs(h_e_k) == 0:
                # 避免除以零
                b_unconstrained = 0
            else:
                b_unconstrained = h_e_k.conj() / (eta_new * K * np.abs(h_e_k) ** 2)
            if np.abs(b_unconstrained) ** 2 <= P_k[k]:
                b_new[k] = b_unconstrained
            else:
                b_new[k] = np.sqrt(P_k[k]) * np.exp(1j * np.angle(h_e_k.conj()))

        # 第三步：优化omega_n
        # 调用传入的优化器函数，确保参数名称匹配
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
            h=h,  # 信道矩阵
            g=g,  # 信道向量
            eta=eta_new,  # 优化后的eta
            b=b_new,  # 优化后的b
            omega=omega_new,  # 优化后的omega
            sigma_a2=sigma_a2,  # 噪声方差
            alpha_min=alpha_min,  # 最小反射系数
            phi=phi,  # 相位偏移
            gamma_param=gamma_param  # gamma参数
        )
        # 检查收敛性
        if iter_num > 0 and np.abs(mse- mse_list[-1]) < tol:
            break

        mse_list.append(mse)


        # 更新变量，进入下一次迭代
        eta = eta_new
        b = b_new
        omega = omega_new

    # 返回优化后的变量和MSE列表
    return eta, b, omega, mse_list
