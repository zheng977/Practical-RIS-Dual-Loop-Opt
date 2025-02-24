# optimizers.py
# -*- coding: utf-8 -*-
"""
不同优化器的实现，用于优化omega_n。
"""
import numpy as np
import cvxpy as cp
from deap import base, creator, tools, algorithms
import random
from pyswarms.single.global_best import GlobalBestPSO
import torch
import torch.nn as nn
from train_nn import PhaseOptimizationNN
from ao_algorithm import ao_algorithm

def optimize_omega_sca(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2, tau=1e-4):
    """
    使用顺序凸近似（SCA）方法优化omega_n，加入正则化项以提升性能。

    参数：
    - h (numpy.ndarray): 用户到RIS的信道矩阵，形状为 (N, K)
    - g (numpy.ndarray): RIS到AP的信道向量，形状为 (N,)
    - eta (float): 当前的eta值
    - b (numpy.ndarray): 当前的b向量，形状为 (K,)
    - omega_old (numpy.ndarray): 上一轮迭代的omega值，形状为 (N,)
    - alpha_min (float): RIS反射系数的最小值
    - phi (float): 相位偏移参数
    - gamma_param (float): 控制函数曲线陡度的参数
    - sigma_a2 (float): AP端噪声方差
    - tau (float): 正则化参数，默认值为1e-3

    返回：
    - omega_new (numpy.ndarray): 更新后的omega值，形状为 (N,)
    """
    K = h.shape[1]  # 用户数量
    N = h.shape[0]  # RIS元素数量

    # 计算alpha_n和其关于omega_n的导数
    sin_term = (np.sin(omega_old - phi) + 1) / 2
    alpha_old = (1 - alpha_min) * sin_term ** gamma_param + alpha_min
    d_alpha = (1 - alpha_min) * gamma_param * sin_term ** (gamma_param - 1) * (np.cos(omega_old - phi) / 2)

    # 预先计算常数
    e_jomega_old = np.exp(1j * omega_old)  # 形状为 (N,)
    c_nk = g.conj()[:, np.newaxis] * h      # 形状为 (N, K)

    # 初始化C_k和D_k
    C_k = np.zeros(K, dtype=complex)        # 形状为 (K,)
    D_k = np.zeros((K, N), dtype=complex)   # 形状为 (K, N)

    for k in range(K):
        A_n = alpha_old * e_jomega_old       # 形状为 (N,)
        B_n = (d_alpha + 1j * alpha_old) * e_jomega_old  # 形状为 (N,)

        C_k[k] = eta * b[k] * np.sum(c_nk[:, k] * A_n)  # 标量
        D_k[k, :] = eta * b[k] * c_nk[:, k] * B_n      # 形状为 (N,)

    # 定义优化变量
    omega = cp.Variable(N)

    # 构建目标函数
    obj = 0
    for k in range(K):
        E_k = C_k[k] - 1 / K
        F_n = D_k[k, :]  # 形状为 (N,)

        # 实部和虚部分别计算
        real_part = cp.real(E_k + F_n @ (omega - omega_old))
        imag_part = cp.imag(E_k + F_n @ (omega - omega_old))

        obj += real_part ** 2 + imag_part ** 2

    # 加入正则化项
    obj += (tau / 2) * cp.sum_squares(omega - omega_old)

    # 定义约束条件
    constraints = [omega >= -np.pi, omega <= np.pi]

    # 形成并求解优化问题
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # 获取优化后的omega值
    omega_new = omega.value

    return omega_new


def optimize_omega_sca_2nd_bound(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2, tau=1e-4):
    """
    使用二阶泰勒展开的凸上界替代函数优化相位omega_n。
    """
    K = h.shape[1]  # 用户数量
    N = h.shape[0]  # RIS元素数量

    # 计算alpha_n和其导数
    sin_term = (np.sin(omega_old - phi) + 1) / 2
    alpha_old = (1 - alpha_min) * sin_term ** gamma_param + alpha_min
    d_alpha = (1 - alpha_min) * gamma_param * sin_term ** (gamma_param - 1) * (np.cos(omega_old - phi) / 2)

    # 计算二阶导数
    d2_alpha = (1 - alpha_min) * gamma_param * (
            (gamma_param - 1) * sin_term ** (gamma_param - 2) * (np.cos(omega_old - phi) / 2) ** 2 -
            sin_term ** (gamma_param - 1) * (np.sin(omega_old - phi) / 2)
    )

    # 预先计算常数
    e_jomega_old = np.exp(1j * omega_old)
    c_nk = g.conj()[:, np.newaxis] * h

    # 计算Lipschitz常数
    L = np.max(np.abs(g)) * np.max(np.abs(h)) * np.max(np.abs(b)) * eta
    L *= (1 - alpha_min) * gamma_param

    # 初始化系数
    C_k = np.zeros(K, dtype=complex)
    D_k = np.zeros((K, N), dtype=complex)
    Q_k = np.zeros((K, N), dtype=float)  # 注意：这里改为实数

    for k in range(K):
        # 零阶项
        A_n = alpha_old * e_jomega_old
        C_k[k] = eta * b[k] * np.sum(c_nk[:, k] * A_n)

        # 一阶项
        B_n = (d_alpha + 1j * alpha_old) * e_jomega_old
        D_k[k, :] = eta * b[k] * c_nk[:, k] * B_n

        # 二阶项上界（使用实数）
        d2u_n = (d2_alpha + 2j * d_alpha - alpha_old) * e_jomega_old
        L_k = np.abs(eta * b[k] * c_nk[:, k] * d2u_n)
        Q_k[k, :] = np.minimum(L * np.ones(N), L_k)

    # 定义优化变量
    omega = cp.Variable(N)
    delta = omega - omega_old

    # 构建目标函数
    obj = 0
    for k in range(K):
        # 分离实部和虚部
        # 零阶项
        E_k_real = np.real(C_k[k] - 1 / K)
        E_k_imag = np.imag(C_k[k] - 1 / K)

        # 一阶项
        F_n_real = np.real(D_k[k, :])
        F_n_imag = np.imag(D_k[k, :])

        # 构造实部表达式
        expr_real = E_k_real + F_n_real @ delta
        expr_imag = E_k_imag + F_n_imag @ delta

        # 添加二阶上界项
        for n in range(N):
            expr_real += (Q_k[k, n] / 2) * (delta[n] ** 2)
            expr_imag += (Q_k[k, n] / 2) * (delta[n] ** 2)

        # 添加到目标函数
        obj += expr_real ** 2 + expr_imag ** 2

    # 添加正则化项
    obj += (tau / 2) * cp.sum_squares(delta)

    # 信任域和相位约束
    trust_radius = 0.1 * np.pi
    constraints = [
        omega >= -np.pi,
        omega <= np.pi,
        cp.norm(delta) <= trust_radius
    ]

    # 求解问题
    try:
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status != cp.OPTIMAL:
            print(f"Warning: Problem status is {prob.status}")
            return omega_old

        return omega.value

    except Exception as e:
        print(f"Optimization error: {str(e)}")
        return omega_old

def ga_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2,
                     population_size=80, num_generations=100, crossover_rate=0.8, mutation_rate=0.2):
    K = h.shape[1]
    N = h.shape[0]
    P_k = np.ones(K) * 1  # 每个用户的发射功率约束，假设都为1

    # 创建DEAP类
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # 适应度函数
    def fitness(individual):
        omega = np.array(individual)
        # 固定omega，执行eta和b的优化
        _, _, _, mse_list = ao_algorithm(
            h=h,
            g=g,
            eta_init=eta,  # 使用当前eta作为初始值
            b_init=b,      # 使用当前b作为初始值
            omega_init=omega,  # 使用候选的omega
            P_k=P_k,
            sigma_a2=sigma_a2,
            alpha_min=alpha_min,
            phi=phi,
            gamma_param=gamma_param,
            optimize_omega_func=no_optimize_omega,  # 使用no_optimize_omega以固定omega
            max_iter=10,
            tol=1e-6
        )
        return (mse_list[-1],)  # 返回优化后的MSE

    # 设置DEAP框架
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -np.pi, np.pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法操作
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=np.pi/10, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 初始化种群
    pop = toolbox.population(n=population_size)

    # 应用遗传算法
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate,
                                 ngen=num_generations, verbose=False)

    # 获取适应度最优的个体
    best_ind = tools.selBest(pop, 1)[0]
    omega_new = np.array(best_ind)

    return omega_new

def pso_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2,
                      num_particles=30, max_iterations=100, w=0.7, c1=1.5, c2=1.5):
    """
    使用粒子群优化（PSO）优化omega_n
    """
    K = h.shape[1]
    N = h.shape[0]
    P_k = np.ones(K) * 1

    # 定义新的适应度函数
    def fitness_function(x):
        # x.shape = (num_particles, N)
        mse = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            omega = x[i]
            # 对每个粒子位置执行eta和b的优化
            _, _, _, mse_history = ao_algorithm(
                h=h,
                g=g,
                eta_init=eta,
                b_init=b,
                omega_init=omega,  # 使用当前粒子位置作为omega
                P_k=P_k,
                sigma_a2=sigma_a2,
                alpha_min=alpha_min,
                phi=phi,
                gamma_param=gamma_param,
                optimize_omega_func=no_optimize_omega,  # 固定omega
                max_iter=10,  # 使用较小的迭代次数优化eta和b
                tol=1e-6
            )
            mse[i] = mse_history[-1]  # 使用优化后的MSE值
        return mse

    # PSO参数
    options = {'c1': c1, 'c2': c2, 'w': w}

    # 定义搜索空间边界
    lower_bounds = -np.pi * np.ones(N)
    upper_bounds = np.pi * np.ones(N)
    bounds = (lower_bounds, upper_bounds)

    # 初始化PSO优化器
    optimizer = GlobalBestPSO(
        n_particles=num_particles,
        dimensions=N,
        options=options,
        bounds=bounds
    )

    # 运行PSO优化
    cost, pos = optimizer.optimize(
        fitness_function,
        iters=max_iterations
    )

    # 返回最佳位置(最优相位配置)
    omega_new = pos

    return omega_new

def no_optimize_omega(h, g, eta, b, omega_old, alpha_min, phi, gamma_param, sigma_a2):
    """
    不优化 omega_n，直接返回旧的 omega。

    参数：
    - h (numpy.ndarray): 用户到RIS的信道矩阵，形状为 (N, K)
    - g (numpy.ndarray): RIS到AP的信道向量，形状为 (N,)
    - eta (float): 当前的eta值
    - b (numpy.ndarray): 当前的b向量，形状为 (K,)
    - omega_old (numpy.ndarray): 上一轮迭代的omega值，形状为 (N,)
    - alpha_min (float): RIS反射系数的最小值
    - phi (float): 相位偏移参数
    - gamma_param (float): 控制函数曲线陡度的参数
    - sigma_a2 (float): AP端噪声方差

    返回：
    - omega_new (numpy.ndarray): 不变的omega值，形状为 (N,)
    """
    return omega_old.copy()

