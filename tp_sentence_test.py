import numpy as np
import tensorflow as tf
import cnnmodel
def test_sentence_level(save_path,name):
    test_word = np.load("./cnndata/cnn_tp_test_word.npy")
    test_pos1 = np.load("./cnndata/cnn_tp_test_pos1.npy")
    test_pos2 = np.load("./cnndata/cnn_tp_test_pos2.npy")
    test_y = np.load("./cnndata/cnn_tp_test_y.npy")
    with open('origin_data/tp_data.txt','r',encoding= 'utf-8') as input:
        test_data = input.readlines()
    no_na = {}
    i = 0
    for line in test_data:
        line = line.strip()
        items = line.split('\t')
        relation = items[2]
        if relation != "NA":
            if relation not in no_na.keys():
                no_na[relation] = []
                no_na[relation].append(i)
            else:
                no_na[relation].append(i)
        i += 1
    test_word_l = []
    test_pos1_l = []
    test_pos2_l = []
    test_y_l = []
    print(no_na.keys())
    for key in no_na.keys():
        test_word_l.append([test_word[x] for x in no_na[key]])
        test_pos1_l.append([test_pos1[x] for x in no_na[key]])
        test_pos2_l.append([test_pos2[x] for x in no_na[key]])
        test_y_l.append([test_y[x] for x in no_na[key]])
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            interact = cnnmodel.interaction(sess, save_path)
    relation = []
    relation_pred = []
    for batch in range(len(test_y_l)):
        batch_word = test_word_l[batch]
        batch_pos1 = test_pos1_l[batch]
        batch_pos2 = test_pos2_l[batch]
        batch_y = test_y_l[batch]
        batch_relation = [np.argmax(i) for i in batch_y]
        batch_relation_pred, batch_prob = interact.test(batch_word, batch_pos1, batch_pos2)
        relation.append(batch_relation)
        relation_pred.append(batch_relation_pred)
    p = []
    r = []
    t_n = 0
    for i in range(len(relation)):
        TP = 0
        FP = 0
        for j in range(len(relation[i])):
            if relation_pred[i][j]!= 0:
                if relation[i][j] == relation_pred[i][j]:
                    TP += 1
                    t_n+= 1
                else:
                    FP += 1
        if (TP+FP)!=0:
            p.append(TP/(TP+FP))
        else:
            p.append(0)
        r.append(TP/(len(relation[i])))
    macro_p = 0
    macro_r = 0
    num = 0
    for i in range(len(relation)):
        num += len(relation[i])
        macro_p+=p[i]
        macro_r+=r[i]
    macro_p /= len(p)
    macro_r /= len(r)
    if (macro_p+macro_r)!=0:
        macro_f1 = 2*(macro_p*macro_r)/(macro_p+macro_r)
    else:
        macro_f1 = 0
    acc = t_n/num
    print(p)
    print(r)
    print(name,acc)
    return acc,[p,r]



if __name__ == "__main__":
    t_max_1 = 0
    t_max_1_p_r = []
    for i in range(1,51):
        print("-------------------")
        print(i)
        max_1,max_1_p_r = test_sentence_level(save_path="./model/" + str(i) + "_11_6_fraction_0.5_cnn_model.ckpt",name = "our_0.5")
        if max_1 > t_max_1:
            t_max_1 = max_1
            t_max_1_p_r = max_1_p_r
    print("t_max_1",t_max_1)
