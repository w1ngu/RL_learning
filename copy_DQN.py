import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
import keras
from keras.optimizers import RMSprop ,adam
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)


class DQN:
    def __init__(
            self,
            n_actions,
            n_features,           #observ  参数个数
            learning_rate=0.01,
            reward_decay=0.9,      #折扣因子
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter      #隔多少步 更换m1参数
        self.memory_size = memory_size                      #记忆库容量大小
        self.batch_size = batch_size                        #随机梯度下降
        self.epsilon_increment = e_greedy_increment      #不断缩小随机的范围
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0                     #记录学习了多少步

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))     #记忆库大小     n_features * 2 + 2 = observation(n_features), action, reward, observation(n_features)

        # print('memory',self.memory_size,self.memory.shape)

        self._build_net()

        # self.sess = tf.Session()
        # if output_graph:
        #     tf.summary.FileWriter("logs/",self.sess.graph)
        # self.sess.run(tf.global_variables_initializer())
        self.loss_his = []
        self.acc_his = []
        # self.his = []

    def target_replace_op(self):           #更新参数
        v1 = self.model2.get_weights()      #m2的参数放到m1
        self.model1.set_weights(v1)
        print("params has changed")

    def _build_net(self):        #构建网络
        # 构建evaluation网络

        eval_inputs = Input(shape=(self.n_features,))  #输入维度
        x = Dense(64, activation='relu')(eval_inputs)   #64个全连接层
        x = Dense(64, activation='relu')(x)              #64个全连接层
        self.q_eval = Dense(self.n_actions)(x)           #n个动作输出

        # 构建target网络，注意这个target层输出是q_next而不是，算法中的q_target
        target_inputs = Input(shape=(self.n_features,)) #输入维度
        x = Dense(64, activation='relu')(target_inputs) #64个全连接层
        x = Dense(64, activation='relu')(x)             #64个全连接层
        self.q_next = Dense(self.n_actions)(x)          #n个动作输出

        self.model1 = Model(target_inputs, self.q_next)
        self.model2 = Model(eval_inputs, self.q_eval)
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

        self.model1.summary()
        self.model2.summary()

    def store_transition(self, s, a, r, s_):        #存储记忆  observe， action, reward,  observe
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # memory是一个二维列表

        self.memory_counter += 1

    def choose_action(self, observation):               #选择下一个动作  输入观测值  返回下一个动作
        print('ovs 1',observation)
        observation = np.array(observation)                #
        observation = observation[np.newaxis, :]         #多一个维度
        print('ovs 2',observation)

        if np.random.uniform() < self.epsilon:           # 根据观测值 预测下一步
            actions_value = self.model1.predict(observation)  #预测四个方位的值
            print('predict actions_value',actions_value)
            action = np.argmax(actions_value)               #选择四个方位的值
            print('action', action)
        else:                                             # 随机e 探索
            action = np.random.randint(0, self.n_actions)
            print('action', action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:    #  更新m1参数
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:   #调用记忆 随机抽取记忆
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.model1.predict(batch_memory[:, -self.n_features:]), self.model2.predict(
            batch_memory[:, :self.n_features])


        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)   # q_target = reward + gamma * Max(q_next)


        hist = self.model2.fit(batch_memory[:, :self.n_features], q_target, epochs=10)


        self.loss_his.append(hist.history['loss'][-1])
        self.acc_his.append(hist.history['acc'][-1])
        # for i in range(0,len(hist.history['loss'])):
        #     self.loss_his.append(hist.history['loss'][i])
        #     self.acc_his.append(hist.history['acc'][i])
        # self.his = hist.history

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_show(self):
        plt.plot(np.arange(len(self.loss_his)),     self.loss_his, color='g')
        plt.plot(np.arange(len(self.acc_his)),      self.acc_his,  color='b')
        print('loss',self.loss_his)
        print('acc_his', self.acc_his)
        plt.show()

        # with open('hist.txt', 'w') as f:
        #     f.write(str(self.his))



