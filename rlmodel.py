import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import cnnmodel
import random
import tqdm
from bert_serving.client import BertClient
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class agent():
    def __init__(self, lr):
        self.len_sentence = 768

        self.cls = tf.placeholder(dtype=tf.float32, shape=[None, self.len_sentence], name='cls')
        self.prob = tf.reshape(layers.fully_connected(self.cls,1,tf.nn.sigmoid),[-1])

        #compute loss
        self.reward_holder = tf.placeholder(shape=[], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        #the probability of choosing 0 or 1
        self.pi  = self.action_holder * (self.prob + 1e-8) + (1 - self.action_holder) * (1 - self.prob+1e-8)

        #loss
        self.loss = -tf.reduce_sum(tf.log(self.pi) * self.reward_holder)

        # minimize loss
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

        self.tvars = tf.trainable_variables()


        #compute gradient
        self.gradients = tf.gradients(self.loss, self.tvars)

        #update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))



def selected_list(list_action):
    l = []
    for i in range(len(list_action)):
        if(list_action[i] == 1):
            l.append(i)
    return l

def get_action(prob):
    key = random.random()
    if(key <= prob):
        return 1
    else:
        return 0

def select_prob(prob_list):
    list_action = [get_action(prob) for prob in prob_list]
    return list_action

def bert_vec(data):
    cls_list = []
    bc = BertClient()
    vec = bc.encode(data)
    for i in range(len(data)):
        cls_list.append(vec[i])
    return cls_list

def read_data(filename):
    with open(filename,'r',encoding = 'utf-8') as f:
        data = [x.strip() for x in f.readlines()]
        return data

def rank(prob_list,batch_size):
    list_action = []
    prob_list.tolist()
    sort_prob = sorted(prob_list,reverse=True)
    flag = sort_prob[int(batch_size*0.5)]
    for i in range(len(prob_list)):
        if(prob_list[i]>=flag):
            list_action.append(1)
        else:
            list_action.append(0)
    return list_action

def random_choose(prob_list):
    list_action = []
    prob_list.tolist()
    for i in range(len(prob_list)):
        key = random.random()
        if(key>0.5):
            list_action.append(1)
        else:
            list_action.append(0)
    return list_action


def train():
    #导入数据
    cls_vec = np.load("./data/cls_vec_0.9.npy")
    cnn_train_word_09 = np.load("./cnndata/cnn_train_0.9_word.npy")
    cnn_train_pos1_09 = np.load("./cnndata/cnn_train_0.9_pos1.npy")
    cnn_train_pos2_09 = np.load("./cnndata/cnn_train_0.9_pos2.npy")
    cnn_train_y_09 = np.load("./cnndata/cnn_train_0.9_y.npy")
    print(cnn_train_word_09.shape)
    print(cnn_train_y_09.shape)

    #开始训练
    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl)
    with g_rl.as_default():
        with sess2.as_default():

            myAgent = agent(0.005)
            num_epoch = 1 #整个rlmodel训练的轮次
            sampletimes = 1 #sampletimes的引入是为了后面baseline 的扩充
            batch_size = 5120 #每次从原数据集中抽取大小为N的批度进行挑选
            f1 = 0
            reward = 0
            T = 4
            max_f1 = 0
            f1_list = [0]
            steps = 1

            #初始化rlmodel
            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver(max_to_keep=300)
            # 用于存储最好的模型参数
            tvars_best = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best):
                tvars_best[index] = var * 0
            # 将梯度缓存p个steps后更新一次
            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0


            g_rl.finalize()
            # 进行num_epoch轮
            for epoch in range(num_epoch):
                print("epoch"+str(epoch))
                all_list = list(range(cnn_train_y_09.shape[0]))
                # 每一轮迭代steps次
                # 相当于选取总的数据集中的某个batch去训练Agent
                for step in range(steps):
                    l_now = time.time()
                    print('    steps:' + str(step))
                    batch = random.sample(all_list, batch_size)
                    batch_cnn_train_word = cnn_train_word_09[batch]
                    batch_cnn_train_pos1 = cnn_train_pos1_09[batch]
                    batch_cnn_train_pos2 = cnn_train_pos2_09[batch]
                    batch_train_y = cnn_train_y_09[batch]
                    batch_train_cls = [cls_vec[x] for x in batch]

                    for j in range(sampletimes):
                        feed_dict = {}
                        feed_dict[myAgent.cls] = batch_train_cls
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        list_action = select_prob(prob)
                        '''
                        list_action = random_choose(prob)
                        '''
                        selected_indexs = selected_list(list_action)
                        print("        "+str(len(selected_indexs)))
                        print("        "+str(prob))

                    #用筛选出来的数据去训练CNN，得到在valid上的loss
                    selected_word = [batch_cnn_train_word[x] for x in selected_indexs]
                    selected_pos1 = [batch_cnn_train_pos1[x] for x in selected_indexs]
                    selected_pos2 = [batch_cnn_train_pos2[x] for x in selected_indexs]
                    selected_y = [batch_train_y[x] for x in selected_indexs]
                    new_f1 = cnnmodel.train_union(selected_word,selected_pos1,selected_pos2,selected_y,save_path='model/batch_cnn_model.ckpt', tag=1)
                    print(new_f1)

                    if(step < T):
                        f1_list.append(new_f1)
                    elif(step == T):
                        reward = 0
                        f1 = sum(f1_list)/T
                    else:
                        reward = new_f1 - f1
                        f1 = (1-1/T)*f1 + 1/T*new_f1
                    print(reward)
                    print(f1)
                    #word pos1 pos2 entity1 entity2 label action reward 这些数据已经准备就绪，开始训练rlmodel
                    feed_dict[myAgent.cls] = [batch_train_cls[x] for x in range(batch_size)]
                    feed_dict[myAgent.reward_holder] = reward
                    feed_dict[myAgent.action_holder] = [list_action[x] for x in range(batch_size)]
                    sess2.run(myAgent.train_op, feed_dict=feed_dict)
                    if (f1 > max_f1):
                        max_f1 = f1
                        saver.save(sess2, save_path='rlmodel/' +"11_6_1_" + 'best_f1_rl_model.ckpt')
                    now = time.time()
                    print(now - l_now)
                    saver.save(sess2, save_path='rlmodel/' +"11_6_1_" + 'rl_model.ckpt')

if __name__ == "__main__":
    train()






