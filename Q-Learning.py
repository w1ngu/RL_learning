import numpy as np
import random

Q = np.zeros((6,2))
R = [0,0,0,0,1,0]


# print(Q)

low = 0
high = 5
alpha = 0.1
gamma = 0.9
e = 0.9

def evc_next(s,a):
    t = [-1,1]
    s = s + t[a]
    if is_not_exist(s):
        return s,0
    return s,R[s]


def choose_action(s):    #状态s下  返回动作 a
    if random.random()>e or (Q[s,:].any() == 0):
        return random.choice([0, 1])
    else:
        return Q[s,:].argmax()

def is_not_exist(s):
    if s<0 or s>5:
        return 1
    else:
        return 0


t = [-1,1]

for i in range(10000):
    s = random.randint(0,5)
    count = 0
    while(count<15):
        a = choose_action(s)
        s_,r = evc_next(s,a)
        if is_not_exist(s_):
            continue

        q_predict = Q[s,a]
        q_target = 0
        if s_==4:
            q_target = r
        else:
            q_target = r + gamma * Q[s_,:].max()
        # q_target = r + b * maxQ(s2)


        # print(s,a,r,s_)
        Q[s,a] = Q[s,a] + alpha*(q_target - q_predict)

        s = s_
        count +=1
    # print('count',count)

print(Q)



