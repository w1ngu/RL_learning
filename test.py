# import gym
# env = gym.make('MountainCar-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

import numpy as np
a = [0,1,0]
a = np.array(a)

if a.any() == 0:
    print(1)