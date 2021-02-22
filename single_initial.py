import numpy as np
import os
import pickle
import random
import json

# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def load_relation2id():
    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()
    return relation2id

def load_word2id():
    print(' reading word embedding data...')
    vec = []
    word2id = {}
    # import the word vec
    # 建立word2id的字典
    f = open('./origin_data/vec.txt', encoding='utf-8')
    info = f.readline()
    print('word vec info:', info)
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    #随机生成‘UNK'以及’BLANK‘的向量并补充进word2vec中
    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)
    np.save('./data/vec.npy', vec)
    return word2id

def initial_sentence(relation2id,word2id,path_name,types):
    print(relation2id)
    fixlen = 70
    f = open(path_name, 'r', encoding='utf-8')
    train_sen = []
    train_ans = []
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = relation2id[content[2]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        train_ans.append(label)

        sentence = content[3:-1]
        en1pos = 0
        en2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        for i in range(min(fixlen, len(sentence))):
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word
        train_sen.append(output)
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for tmp in train_sen:
        tmp_train_word = []
        tmp_train_pos1 = []
        tmp_train_pos2 = []
        for i in tmp:
            tmp_train_word.append(i[0])
            tmp_train_pos1.append(i[1])
            tmp_train_pos2.append(i[2])
        train_word.append(tmp_train_word)
        train_pos1.append(tmp_train_pos1)
        train_pos2.append(tmp_train_pos2)
    np.save("./data/train_x.npy", train_sen)
    np.save("./cnndata/cnn_"+types+"_word.npy",train_word)
    np.save("./cnndata/cnn_"+types+"_pos1.npy",train_pos1)
    np.save("./cnndata/cnn_"+types+"_pos2.npy",train_pos2)
    np.save("./cnndata/cnn_"+types+"_y.npy", train_ans)

relation2id = load_relation2id()
word2id = load_word2id()
initial_sentence(relation2id,word2id,"origin_data/ha1_test1.txt","tp_test")
