from RL_brain import DQN
import gym


def run_maze():
    env = gym.make('MountainCar-v0')
    env.reset()

    # step = 0
    #
    # for episode in range(300):
    #     # initial observation
    #     observation = env.reset()             #得到起初的观测值
    #
    #     while True:
    #         # fresh env
    #         env.render()
    #
    #         # RL choose action based on observation
    #         action = RL.choose_action(observation)        #选择下一个动作
    #
    #         # RL take action and get next observation and reward
    #         observation_, reward, done = env.step(action)    #根据下一个动作得到 下一个观测值， 奖励， 是否结束标志
    #
    #         RL.store_transition(observation, action, reward, observation_)     #存储记忆   观测值， 动作， 得到的奖励， 下一个观测值
    #
    #         if (step > 200) and (step % 5 == 0):    #  如果超过200 步 是5的倍数
    #             RL.learn()                          # 学习
    #
    #         # swap observation
    #         observation = observation_              #更新观测值
    #
    #         # break while loop when end of this episode
    #         if done:                                #结束标志
    #             break
    #         step += 1                               #步数累加
    #
    # # end of game
    # print('game over')
    # env.destroy()


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    print(env.action_space,env.action_space.n)
    print(env.observation_space.shape[0])
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.n_features)

    observation = env.reset()  # 得到起初的观测值

    print(observation)


    #
    # RL = DQN(env.n_actions, env.n_features,  # import DQN
    #          learning_rate=0.01,
    #          reward_decay=0.9,
    #          e_greedy=0.9,
    #          replace_target_iter=10,
    #          memory_size=4000,
    #          output_graph=True
    #          )
    #
    # for episode in range(300):
    #     # initial observation
    #     observation = env.reset()  # 得到起初的观测值
    #
    #     print(observation)


