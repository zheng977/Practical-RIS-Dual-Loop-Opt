# -*- coding: utf-8 -*-
"""
主程序，执行交替优化（AO）算法，比较不同优化方法（如SCA、GA、PSO、神经网络和不优化）的效果。
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import squeeze
from channel_simulation import generate_channels_new
from ao_algorithm import ao_algorithm
from optimizers import  ga_optimize_omega, pso_optimize_omega, no_optimize_omega, optimize_omega_sca,optimize_omega_sca_2nd_bound
import torch
from location import location


def verify_mse_calculation(h, g, eta, b, omega, sigma_a2, alpha_min, phi, gamma_param, K):
    """
    详细验证MSE的计算过程，打印中间结果。

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
    - K: 用户数量

    返回:
    - mse: 计算得到的MSE值
    """
    # 1. 计算反射系数alpha
    alpha = (1 - alpha_min) * ((np.sin(omega - phi) + 1) / 2) ** gamma_param + alpha_min
    print("\n1. 反射系数alpha:")
    print(f"alpha范围: [{np.min(alpha)}, {np.max(alpha)}]")

    # 2. 构建RIS反射矩阵Φ
    Phi = np.diag(alpha * np.exp(1j * omega))
    print("\n2. RIS反射矩阵Φ:")
    print(f"Φ对角线元素模值范围: [{np.min(np.abs(Phi.diagonal()))}, {np.max(np.abs(Phi.diagonal()))}]")

    # 3. 计算等效信道h_e
    h_e = g.conj().T @ Phi @ h
    print("\n3. 等效信道h_e:")
    print(f"h_e模值范围: [{np.min(np.abs(h_e))}, {np.max(np.abs(h_e))}]")

    # 4. 计算每个用户的信号项
    signal_terms = np.abs(eta * h_e * b - 1 / K) ** 2
    print("\n4. 每个用户的信号项 |ηhe,kbk - 1/K|²:")
    for k in range(K):
        print(f"用户{k + 1}: {signal_terms[k]:.2e}")

    # 5. 计算总信号项
    total_signal = np.sum(signal_terms)
    print(f"\n5. 总信号项 Σ|ηhe,kbk - 1/K|²: {total_signal:.2e}")

    # 6. 计算噪声项
    noise_term = eta ** 2 * sigma_a2
    print(f"\n6. 噪声项 η²σa²: {noise_term:.2e}")

    # 7. 计算最终MSE
    mse = total_signal + noise_term
    print(f"\n7. 最终MSE = 信号项 + 噪声项: {mse:.2e}")

    # 8. 验证功率约束
    power_constraints = np.abs(b) ** 2
    print("\n8. 验证功率约束 |bk|²:")
    for k in range(K):
        print(f"用户{k + 1}功率: {power_constraints[k]:.4f}")

    return mse


# 使用示例:
"""

)
"""

# ====================== 超参数设置 ======================
K = 10 #用户数量
N = 100# RIS 元素数量
sigma_a2 = 1e-8  # AP 端噪声方差
max_iter = 20  # 最大迭代次数
tol = 1e-5  # 收敛阈值
alpha_min = 0.2  # RIS 反射系数的最小值
phi = 0.0  # 相位偏移参数
gamma_param = 3.0  # 控制函数曲线陡度的参数
P_k = np.ones(K)*1  # 每个用户的发射功率约束，假设都为1

# GA参数
population_size = 80  # 种群规模
num_generations = 100  # 迭代代数
crossover_rate = 0.8  # 交叉率
mutation_rate = 0.3  # 变异率

# PSO参数
num_particles = 200  # 粒子数量
max_iterations_pso = 200  # PSO最大迭代次数
w = 0.7  # 惯性权重
c1 = 3.0  # 个体加速因子
c2 = 3.0  # 社会加速因子

# 设置中文字体以避免Matplotlib的字体警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，确保可重复性
np.random.seed(2024)
torch.manual_seed(2024)

pl_RA, pl_UA, dis_RA, dis_UA, location_user = location(K)
location_AP = np.array([-50, 0, 12])
location_RIS = np.array([0, 0, 12])
h, g = generate_channels_new(N, K, pl_RA, pl_UA, location_user, location_AP, location_RIS)
g = g.squeeze()

print(f"信道矩阵 h 的形状：{h.shape}")
print(f"信道向量 g 的形状：{g.shape}")

# 初始化变量
eta_init = 1.0
b_init = np.sqrt(P_k) * np.exp(1j * 2 * np.pi * np.random.rand(K))
omega_init = np.random.uniform(low=-0*np.pi, high=0*np.pi, size=N)

# # GA优化
# print("运行使用 GA 优化 omega_n 的 AO 算法...")
# omega_opt = ga_optimize_omega(h=h, g=g, eta=eta_init, b=b_init, omega_old=omega_init,
#                             alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
#                             sigma_a2=sigma_a2, population_size=population_size,
#                             num_generations=num_generations, crossover_rate=crossover_rate,
#                             mutation_rate=mutation_rate)
#
# print("使用GA找到的最佳相位进行AO优化...")
# eta_opt_ga, b_opt_ga, omega_opt_ga, mse_list_ga = ao_algorithm(h=h, g=g, eta_init=eta_init,
#     b_init=b_init, omega_init=omega_opt, P_k=P_k, sigma_a2=sigma_a2, alpha_min=alpha_min,
#     phi=phi, gamma_param=gamma_param, optimize_omega_func=no_optimize_omega,
#     max_iter=max_iter, tol=tol)

# print(f"GA 优化的最终 MSE: {mse_list_ga[-1]}")

print("运行使用 sca 优化 omega_n 的 AO 算法...")
eta_opt_sca, b_opt_sca, omega_opt_sca, mse_list_sca = ao_algorithm(h=h, g=g,
    eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
    optimize_omega_func=optimize_omega_sca, max_iter=max_iter, tol=tol)

# 打印每次迭代的MSE值
print("\nSCA所有迭代的MSE值：")
for i, mse in enumerate(mse_list_sca):
    print(f"迭代 {i+1}: {mse}")

# 使用你的验证函数计算最终状态的MSE
final_mse = verify_mse_calculation(
    h=h, g=g, eta=eta_opt_sca, b=b_opt_sca, omega=omega_opt_sca,
    sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi,
    gamma_param=gamma_param, K=K
)

# 计算最后一个MSE和验证MSE的相对误差
relative_error = abs(mse_list_sca[-1] - final_mse) / final_mse
print(f"\n最后一次迭代的MSE: {mse_list_sca[-1]}")
print(f"验证计算的MSE: {final_mse}")
print(f"相对误差: {relative_error:.2%}")
# # PSO优化
# print("运行使用 PSO 优化 omega_n 的 AO 算法...")
# omega_opt_pso = pso_optimize_omega(h=h, g=g, eta=eta_init, b=b_init,
#     omega_old=omega_init, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
#     sigma_a2=sigma_a2, num_particles=num_particles, max_iterations=max_iterations_pso,
#     w=w, c1=c1, c2=c2)
#
# print("使用PSO找到的最佳相位进行AO优化...")
# eta_opt_pso, b_opt_pso, omega_opt_pso, mse_list_pso = ao_algorithm(h=h, g=g,
#     eta_init=eta_init, b_init=b_init, omega_init=omega_opt_pso, P_k=P_k,
#     sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
#     optimize_omega_func=no_optimize_omega, max_iter=max_iter, tol=tol)
#
# print(f"PSO 优化的最终 MSE: {mse_list_pso[-1]}")
#
# # 不优化
# print("运行不优化 omega_n 的 AO 算法...")
# eta_opt_no, b_opt_no, omega_opt_no, mse_list_no = ao_algorithm(h=h, g=g,
#     eta_init=eta_init, b_init=b_init, omega_init=omega_init, P_k=P_k,
#     sigma_a2=sigma_a2, alpha_min=alpha_min, phi=phi, gamma_param=gamma_param,
#     optimize_omega_func=no_optimize_omega, max_iter=max_iter, tol=tol)
#
# print(f"不优化 omega 的最终 MSE: {mse_list_no[-1]}")
#
# # 绘制MSE随迭代次数的变化曲线
# plt.figure(figsize=(12, 8))
# plt.plot(range(1, len(mse_list_ga) + 1), [x * 1000 for x in mse_list_ga], 'r-o', label='GA 优化')
# plt.plot(range(1, len(mse_list_sca) + 1), [x * 1000 for x in mse_list_sca], 'b-s', label='SCA 优化')
# plt.plot(range(1, len(mse_list_pso) + 1), [x * 1000 for x in mse_list_pso], 'g-^', label='PSO 优化')
# plt.plot(range(1, len(mse_list_no) + 1), [x * 1000 for x in mse_list_no], 'k--', label='不优化 omega')
# plt.xlabel('迭代次数', fontsize=14)
# plt.ylabel('MSE (x $10^{-3}$)', fontsize=14)
# plt.title('MSE 随迭代次数的变化', fontsize=16)
# plt.grid(True)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()
#
# # 输出优化结果
# print("\n使用 GA 优化的结果：")
# print("优化后的 eta 值：", eta_opt_ga)
# print("优化后的 b_k 值：", b_opt_ga)
# print("优化后的 omega_n 值：", omega_opt_ga)
#
# print("\n使用 SCA 优化的结果：")
# print("优化后的 eta 值：", eta_opt_sca)
# print("优化后的 b_k 值：", b_opt_sca)
# print("优化后的 omega_n 值：", omega_opt_sca)
#
# print("\n使用 PSO 优化的结果：")
# print("优化后的 eta 值：", eta_opt_pso)
# print("优化后的 b_k 值：", b_opt_pso)
# print("优化后的 omega_n 值：", omega_opt_pso)
#
# print("\n不优化 omega 的结果：")
# print("优化后的 eta 值：", eta_opt_no)
# print("优化后的 b_k 值：", b_opt_no)
# print("优化后的 omega_n 值：", omega_opt_no)