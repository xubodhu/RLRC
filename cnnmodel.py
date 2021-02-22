import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.len_sentence = 70
        self.num_epochs = 50
        self.num_classes = 53
        self.cnn_size = 230
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        self.word_embedding = 50
        self.keep_prob = 0.5
        self.batch_size = 160
        self.num_steps = 10000
        self.lr= 0.01

class CNN():
    def __init__(self, word_embeddings, setting):

        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes = setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)
        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)

        self.inputs = tf.concat(axis=2, values=[self.input_word_ebd, self.input_pos1_ebd, self.input_pos2_ebd])
        self.inputs = tf.reshape(self.inputs, [-1, self.len_sentence, self.word_embedding + self.pos_size * 2, 1])

        conv = layers.conv2d(inputs=self.inputs, num_outputs=self.cnn_size, kernel_size=[3, 60], stride=[1, 60],
                             padding='SAME')

        max_pool = layers.max_pool2d(conv, kernel_size=[70, 1], stride=[1, 1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh, keep_prob=self.keep_prob)

        self.outputs = layers.fully_connected(inputs=drop, num_outputs=self.num_classes, activation_fn=tf.nn.softmax)
        self.cross_loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(self.outputs), axis=1))
        self.reward = tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())

        self.final_loss = self.cross_loss + self.l2_loss

        # accuracy
        self.pred = tf.argmax(self.outputs, axis=1)
        self.pred_prob = tf.reduce_max(self.outputs, axis=1)

        self.y_label = tf.argmax(self.input_y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y_label), 'float'))

        # minimize loss
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.final_loss)

def train_union(path_word,path_pos1,path_pos2,path_cnn_train_y,save_path,tag):
    wordembedding = np.load('./data/vec.npy')
    with open('./data/valid_0.1.txt','r',encoding= 'utf-8') as input:
        valid_data = input.readlines()
    word = path_word
    pos1 = path_pos1
    pos2 = path_pos2
    cnn_train_y = path_cnn_train_y
    valid_word_01 = np.load("./cnndata/cnn_valid_0.1_word.npy")
    valid_pos1_01 = np.load("./cnndata/cnn_valid_0.1_pos1.npy")
    valid_pos2_01 = np.load("./cnndata/cnn_valid_0.1_pos2.npy")
    settings = Settings()
    settings.num_epochs = 15
    settings.batch_size = 2
    settings.lr = 0.1
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(path_word) // settings.batch_size

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #saver.restore(sess,save_path=save_path)
            for epoch in range(1,settings.num_epochs+1):

                bar = range(settings.num_steps)

                for _ in bar:

                    sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    batch_word = [word[x] for x in sample_list]
                    batch_pos1 = [pos1[x] for x in sample_list]
                    batch_pos2 = [pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_word
                    feed_dict[model.input_pos1] = batch_pos1
                    feed_dict[model.input_pos2] = batch_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    #break
                saver.save(sess, save_path=save_path)
                with open('data/valid_0.1_label_entitypair.pkl', 'rb') as input:
                    label_entitypair = pickle.load(input)
                pred_entitypair = {}
                batch_size = 100
                steps = len(valid_word_01) // batch_size + 1
                for step in range(steps):
                    batch_valid_word = valid_word_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_pos1 = valid_pos1_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_pos2 = valid_pos2_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_date = valid_data[batch_size * step:batch_size * (step + 1)]
                    batch_entitypair = []
                    for line in batch_valid_date:
                        items = line.split('\t')
                        e1 = items[0]
                        e2 = items[1]
                        batch_entitypair.append(e1 + '$' + e2)
                    feed_dict = {}
                    feed_dict[model.input_word] = batch_valid_word
                    feed_dict[model.input_pos1] = batch_valid_pos1
                    feed_dict[model.input_pos2] = batch_valid_pos2
                    feed_dict[model.keep_prob] = 1
                    batch_relation, batch_prob = sess.run([model.pred,model.pred_prob], feed_dict=feed_dict)

                    assert (len(batch_relation) == len(batch_prob) and len(batch_relation) == len(batch_entitypair))
                    for i in range(len(batch_relation)):
                        if batch_relation[i] != 0:
                            tmp_key = batch_entitypair[i]
                            tmp_value = (batch_prob[i], batch_relation[i])
                            if tmp_key not in pred_entitypair.keys():
                                pred_entitypair[tmp_key] = []
                                pred_entitypair[tmp_key] = tmp_value
                            elif tmp_value[0] > pred_entitypair[tmp_key][0]:
                                pred_entitypair[tmp_key] = tmp_value
                list_pred = []
                for key in pred_entitypair.keys():
                    tmp_prob = pred_entitypair[key][0]
                    tmp_relation = pred_entitypair[key][1]
                    tmp_entitypair = key
                    list_pred.append((tmp_prob, tmp_entitypair, tmp_relation))
                list_pred = sorted(list_pred, key=lambda x: x[0], reverse=True)
                true_positive = 0
                for i, item in enumerate(list_pred):
                    tmp_entitypair = item[1]
                    tmp_relation = item[2]
                    label_relations = label_entitypair[tmp_entitypair]
                    if tmp_relation in label_relations:
                        true_positive += 1
                i += 1
                file = open("./data/num_entitypair_true", "r")
                num_entitypair_true = eval(file.read())
                if(i == 0):
                    p = 0
                else:
                    p = float(true_positive/i)
                r = float(true_positive/num_entitypair_true)
                if((p+r)==0):
                    f = 0
                else:
                    f = 2*(p*r)/(p+r)

    return f


def train(path_train_word,path_train_pos1,path_train_pos2,path_train_y,save_path,tag):

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')

    cnn_train_word = np.load(path_train_word)
    cnn_train_pos1 = np.load(path_train_pos1)
    cnn_train_pos2 = np.load(path_train_pos2)
    cnn_train_y    = np.load(path_train_y)
    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // settings.batch_size

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=300)
            #saver.restore(sess,save_path=save_path)
            for epoch in range(1,settings.num_epochs+1):

                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))

                for _ in bar:

                    sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    batch_train_word = [cnn_train_word[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    #break
                saver.save(sess, save_path="./model/"+str(epoch)+save_path)

class interaction():

    def __init__(self,sess,save_path ='model/model.ckpt3'):

        self.settings = Settings()
        wordembedding = np.load('./data/vec.npy')

        self.sess = sess
        with tf.variable_scope("model"):
            self.model = CNN(word_embeddings=wordembedding, setting=self.settings)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,save_path)

    def test(self,batch_test_word,batch_test_pos1,batch_test_pos2):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.keep_prob] = 1
        relation,prob = self.sess.run([self.model.pred,self.model.pred_prob],feed_dict = feed_dict)

        return (relation,prob)


    def save_cnnmodel(self,save_path):
        with self.sess.as_default():
            self.saver.save(self.sess, save_path=save_path)

    def tvars(self):
        with self.sess.as_default():
            tvars = self.sess.run(self.model.tvars)
            return tvars

    def update_tvars(self,tvars_update):
        with self.sess.as_default():
            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)

if __name__ == "__main__":
    buf = "11_6_fraction_0.5_train"
    buf2 = "11_6_fraction_0.5"
    train("./cnndata/cnn_" + buf +  "_word.npy","./cnndata/cnn_" + buf +  "_pos1.npy","./cnndata/cnn_" + buf +  "_pos2.npy","./cnndata/cnn_" + buf +  "_y.npy", "_"+ buf2 +"_cnn_model.ckpt",0)










