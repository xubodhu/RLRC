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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        self.loss = tf.reduce_sum(tf.log(self.pi)) * self.reward_holder

        # minimize loss
        optimizer = tf.train.GradientDescentOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

        self.tvars = tf.trainable_variables()

def read_train():
    data = []
    train_word = np.load('./cnndata/cnn_train_word.npy')
    train_pos1 = np.load('./cnndata/cnn_train_pos1.npy')
    train_pos2 = np.load('./cnndata/cnn_train_pos2.npy')
    y_train = np.load('cnndata/cnn_train_y.npy')
    with open('origin_data/train.txt','r',encoding= 'utf-8') as input:
        train_data = input.readlines()
    for i in range(train_word.shape[0]):
        tmp = (train_word[i],train_pos1[i],train_pos2[i],y_train[i],train_data[i])
        data.append(tmp)
    return data

def bert_vec():
    cls_list = []
    '''
    bc = BertClient()
    for i in range(20):
        b_size = len(data)//20
        vec = bc.encode(data[i*b_size:(i+1)*b_size])
        for j in range(vec.shape[0]):
            cls_list.append(vec[j])
    vec = bc.encode(data[(i+1)*b_size:])
    '''
    vec = np.load("./data/cls.npy")
    for j in range(vec.shape[0]):
        cls_list.append(vec[j])
    return cls_list

def read_data(filename):
    with open(filename,'r',encoding = 'utf-8') as f:
        data = [x.strip() for x in f.readlines()]
        return data

def get_value():
    value = []
    with tf.Session() as sess:
        rl_model = agent(0.01)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, save_path='rlmodel/11_6_1_rl_model.ckpt')
        cls_o = bert_vec()
        feed_dict = {}
        feed_dict[rl_model.cls] = cls_o
        prob = sess.run(rl_model.prob, feed_dict=feed_dict)
        lnum = 0
        for tmp in prob:
            value.append(tmp)
            if(tmp<0.5):
                lnum += 1
        print(lnum)
    print("ok")
    return value

def ins_value():
    data = read_train()
    value = get_value()
    v_dict = {}
    for i in range(len(data)):
        v_dict[(value[i],i)] = data[i]
    print(len(v_dict.keys()))
    return v_dict

def v_dict_sort(v_dict):
    ans = sorted(v_dict.items(), key=lambda item: item[0][0])
    return ans

def base_prob(v_dict):
    data = []
    for prob in v_dict.keys():
        data_buf = v_dict[prob]
        key = random.random()
        if(key <= prob[0]):
            data.append(data_buf)
    return data

def base_high(v_dict,rate):
    data = []
    ans = v_dict_sort(v_dict)
    len_ans = len(ans)
    tag = int((1-rate)*len_ans)
    line = ans[tag][0][0]
    for prob in v_dict.keys():
        data_buf = v_dict[prob]
        if(prob[0]>=line):
            data.append(data_buf)
    return data

def base_low(v_dict,rate):
    data = []
    ans = v_dict_sort(v_dict)
    len_ans = len(ans)
    tag = int((rate)*len_ans)
    line = ans[tag][0][0]
    for prob in v_dict.keys():
        data_buf = v_dict[prob]
        if(prob[0]<=line):
            data.append(data_buf)
    return data

def s_train(data,name):
    train_word = []
    train_pos1 = []
    train_pos2 = []
    train_y = []
    filter_train = []
    for tmp in data:
        train_word.append(tmp[0])
        train_pos1.append(tmp[1])
        train_pos2.append(tmp[2])
        train_y.append(tmp[3])
        filter_train.append(tmp[4])
    np.save("./cnndata/cnn_" + name + "_train_word", train_word)
    np.save("./cnndata/cnn_" + name + "_train_pos1", train_pos1)
    np.save("./cnndata/cnn_" + name + "_train_pos2", train_pos2)
    np.save("./cnndata/cnn_" + name + "_train_y", train_y)
    with open("./origin_data/" + name + "train","w",encoding="utf-8") as f:
        for line in filter_train:
            f.write(line)

if __name__ == "__main__":
    print("produce v_dict......")
    v_dict = ins_value()
    print("get trains......")
    '''
    data_prob = base_prob(v_dict)
    '''
    '''
     s_train(data_prob,"prob")
    '''
    for line in [0.9,0.8,0.7,0.6,0.5,0.4]:

        data_fraction = base_high(v_dict,line)
        print("store trains......")
        '''
        s_train(data_fraction,"9_27_fraction_"+str(line))
        '''
        s_train(data_fraction,"11_6_fraction_"+str(line))





