import numpy as np
import time
import sys

import gym
from  RL_brain import DQN

env = gym.make('CartPole-v0')
print('CartPole-v0')
print(env.action_space,env.action_space.n)
print(env.observation_space.shape[0])                           # 推车位置 车速 杆子角度 杆子末端速度
print(env.observation_space.high)                               # 上界 [ 4.8000002e+00  3.4028235e+38  4.1887903e-01  3.4028235e+38]
print(env.observation_space.low)                                # 下界 [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
print(env.x_threshold, env.theta_threshold_radians)

RL = DQN(n_actions=env.action_space.n,                          # 动作空间 2
         n_features=env.observation_space.shape[0],             # 观测到的 特征4  推车位置 车速 杆子角度 杆子末端速度
         learning_rate=0.01,                                    # 学习速率
         reward_decay=0.9,                                      # 折扣因子
         e_greedy=0.9,                                          # 随机探索率
         replace_target_iter=100,                               # 每隔100代 更新Q_target
         memory_size=2000,                                      # 记忆空间大小
         e_greedy_increment=0.001                               # 探索衰减率
         )

total_step = 0

for i in range(100):
    observation = env.reset()                                   # 重置

    while True:
        env.render()                                            # 重绘环境的一帧
        action = RL.choose_action(observation)                  # 选择下一个动作

        observation_, reward, done, _  = env.step(action)     # 根据下一个动作得到 下一个观测值， 奖励， 是否结束标志

        x, x_dot, theta, theta_dot = observation_               # 推车位置 车速 杆子角度 杆子末端速度

        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8   # (2.4-abs(x)) / (2.4 -0.8)   x 在 0   r1 最大
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.8     # theta = 0 r2 最大
        r = r1+r2

        # print('reward', reward,r)

        RL.store_transition(observation, action, r, observation_)  # 存储记忆   观测值， 动作， 得到的奖励， 下一个观测值


        if total_step>1000:                                     # 如果超过1000 步
            RL.learn()                                          # 学习

        observation = observation_                              # 更新观测值

        if done:                                                # 结束标志
            break
        total_step += 1                                         # 步数累加

RL.plot_show()
RL.save()

