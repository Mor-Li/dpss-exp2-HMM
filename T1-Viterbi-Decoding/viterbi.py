#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Mo Li @ 2024-11-27
# 完整的隐马尔可夫模型（HMM）实现，包括前向算法、后向算法和维特比算法

import numpy as np


class HMM:
    """隐马尔可夫模型（Hidden Markov Model）

    一个包含3个状态和2个观测类别的HMM。

    属性:
        ob_category (list, 长度为2): 观测类别
        total_states (int): 状态数量，默认为3
        pi (array, 形状为(3,)): 初始状态概率
        A (array, 形状为(3, 3)): 状态转移概率矩阵。A.sum(axis=1) 必须全部为1。
                                      A[i, j] 表示从状态i转移到状态j的概率。
        B (array, 形状为(3, 2)): 发射概率矩阵。B.sum(axis=1) 必须全部为1。
                                      B[i, k] 表示状态i发射观测k的概率。
    """

    def __init__(self):
        self.ob_category = ['THU', 'PKU']  # 0: THU, 1: PKU
        self.total_states = 3
        self.pi = np.array([0.2, 0.4, 0.4])  # 初始状态概率
        self.A = np.array([[0.1, 0.6, 0.3],
                           [0.3, 0.5, 0.2],
                           [0.7, 0.2, 0.1]])  # 状态转移概率矩阵
        self.B = np.array([[0.5, 0.5],
                           [0.4, 0.6],
                           [0.7, 0.3]])  # 发射概率矩阵

    def forward(self, ob):
        """HMM 前向算法。

        参数:
            ob (array, 形状为(T,)): (o1, o2, ..., oT), 观测序列

        返回:
            fwd (array, 形状为(T, 3)): fwd[t, s] 表示在给定观测 ob[0:t+1] 的情况下，
                                       在时间步 t 达到状态 s 的所有路径的概率和
            prob: HMM 模型生成观测序列的概率
        """
        T = ob.shape[0]  # 观测序列长度
        fwd = np.zeros((T, self.total_states))  # 初始化前向概率矩阵

        # 初始化阶段：计算时间步0的前向概率
        for s in range(self.total_states):
            fwd[0, s] = self.pi[s] * self.B[s, ob[0]]
            # print(f"前向初始化: fwd[0, {s}] = pi[{s}] * B[{s}, {ob[0]}] = {self.pi[s]} * {self.B[s, ob[0]]} = {fwd[0, s]}")

        # 递推阶段：计算后续时间步的前向概率
        for t in range(1, T):
            for s in range(self.total_states):
                # 计算所有前一状态到当前状态的转移概率之和，并乘以发射概率
                fwd[t, s] = np.sum(fwd[t - 1] * self.A[:, s]) * self.B[s, ob[t]]
                # print(f"前向递推: fwd[{t}, {s}] = sum(fwd[{t-1}, :] * A[:, {s}]) * B[{s}, {ob[t]}] = {fwd[t, s]}")

        prob = fwd[-1, :].sum()  # 观测序列的总概率
        return fwd, prob

    def backward(self, ob):
        """HMM 后向算法。

        参数:
            ob (array, 形状为(T,)): (o1, o2, ..., oT), 观测序列

        返回:
            bwd (array, 形状为(T, 3)): bwd[t, s] 表示在给定观测 ob[t+1::] 的情况下，
                                       在时间步 t 达到状态 s 的所有路径的概率和
            prob: HMM 模型生成观测序列的概率
        """
        T = ob.shape[0]  # 观测序列长度
        bwd = np.zeros((T, self.total_states))  # 初始化后向概率矩阵

        # 初始化阶段：设置时间步T-1的后向概率为1
        bwd[T - 1, :] = 1
        # print(f"后向初始化: bwd[{T-1}, :] = {bwd[T-1, :]}")

        # 递推阶段：计算前一时间步的后向概率
        for t in range(T - 2, -1, -1):
            for s in range(self.total_states):
                # 计算当前状态到所有可能后续状态的转移概率、发射概率与后向概率的乘积之和
                bwd[t, s] = np.sum(self.A[s, :] * self.B[:, ob[t + 1]] * bwd[t + 1, :])
                # print(f"后向递推: bwd[{t}, {s}] = sum(A[{s}, :] * B[:, {ob[t+1]}] * bwd[{t+1}, :]) = {bwd[t, s]}")

        # 计算观测序列的总概率
        prob = np.sum(self.pi * self.B[:, ob[0]] * bwd[0, :])
        return bwd, prob

    def viterbi(self, ob):
        """Viterbi 解码算法。

        参数:
            ob (array, 形状为(T,)): (o1, o2, ..., oT), 观测序列

        返回:
            best_prob: 最佳状态序列的概率
            best_path: 最佳状态序列
        """
        T = ob.shape[0]  # 观测序列长度
        delta = np.zeros((T, self.total_states))  # 存储最大概率
        phi = np.zeros((T, self.total_states), dtype=np.int32)  # 存储路径指针
        best_prob, best_path = 0.0, np.zeros(T, dtype=np.int32)  # 最终结果

        # 初始化阶段：计算时间步0的delta值
        for s in range(self.total_states):
            delta[0, s] = self.pi[s] * self.B[s, ob[0]]
            phi[0, s] = 0  # 初始时没有前驱状态
            # print(f"Viterbi初始化: delta[0, {s}] = pi[{s}] * B[{s}, {ob[0]}] = {delta[0, s]}")

        # 递推阶段：计算后续时间步的delta值和phi指针
        for t in range(1, T):
            for s in range(self.total_states):
                # 计算来自所有前一状态的概率
                prob = delta[t - 1] * self.A[:, s]
                phi[t, s] = np.argmax(prob)  # 记录最大概率对应的前一状态
                delta[t, s] = np.max(prob) * self.B[s, ob[t]]
                # print(f"Viterbi递推: delta[{t}, {s}] = max(delta[{t-1}, :] * A[:, {s}]) * B[{s}, {ob[t]}] = {delta[t, s]}")

        # 终止阶段：找到最后时间步的最大概率和对应的状态
        best_path[T - 1] = np.argmax(delta[T - 1, :])
        best_prob = delta[T - 1, best_path[T - 1]]
        # print(f"Viterbi终止: 最佳概率 = {best_prob}, 最后状态 = {best_path[T-1]}")

        # 回溯阶段：通过phi指针找到最佳路径
        for t in reversed(range(T - 1)):
            best_path[t] = phi[t + 1, best_path[t + 1]]
            # print(f"Viterbi回溯: best_path[{t}] = phi[{t+1}, {best_path[t+1]}] = {best_path[t]}")

        return best_prob, best_path


if __name__ == "__main__":
    # 创建HMM模型实例
    model = HMM()
    # 定义观测序列，0代表THU，1代表PKU
    observations = np.array([0, 1, 0, 1, 1])  # [THU, PKU, THU, PKU, PKU]
    
    # 执行前向算法
    fwd, p_forward = model.forward(observations)
    print("前向概率总和:", p_forward)
    print("前向概率矩阵:\n", fwd)
    
    # 执行后向算法
    bwd, p_backward = model.backward(observations)
    print("后向概率总和:", p_backward)
    print("后向概率矩阵:\n", bwd)
    
    # 执行维特比算法
    prob, path = model.viterbi(observations)
    print("最佳路径概率:", prob)
    print("最佳路径序列:", path)