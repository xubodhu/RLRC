import numpy as np
import os
import pickle
import random



def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def mt(word2id,relation2id,path,filename_x,filename_y):
    fixlen =70
    f = open(path, 'r', encoding='utf-8')
    test_sen = []
    test_ans = []
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        en1 = content[2]
        en2 = content[3]
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        test_ans.append(label)

        sentence = content[5:-1]
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
        test_sen.append(output)

    np.save(filename_x, test_sen)
    np.save(filename_y, test_ans)


def init_data():

    print(' reading word embedding data...')
    vec = []
    word2id = {}
    #建立word2id的字典
    f = open('./origin_data/vec.txt', encoding='utf-8')
    info = f.readline()
    print ('word vec info:',info)
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

    print('reading entity to id ')
    #这里实体对应id的映射，在之前已经构造
    with open('data/dict_entityname2id.pkl','rb') as input:
        dict_entityname2id = pickle.load(input)

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

    # length of sentence is 70
    fixlen = 70

    print("producting train_x")
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')
    train_sen = []
    train_ans = []
    train_entitypair = []
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        en1 = content[2]
        en2 = content[3]
        en1id = dict_entityname2id[en1]
        en2id = dict_entityname2id[en2]
        train_entitypair.append((en1id,en2id))
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        train_ans.append(label)

        sentence = content[5:-1]
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

    train_entitypair = np.array(train_entitypair)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_sen)
    np.save('./cnndata/cnn_train_y.npy', train_ans)
    np.save('./cnndata/cnn_train_entitypair',train_entitypair)

    print('producitng test_x')
    mt(word2id,relation2id,"./origin_data/test.txt","./data/test_x.npy","./cnndata/cnn_test_y.npy")
    print("producting train_0.9_x")
    mt(word2id,relation2id,"./data/train_0.9.txt","./data/train_0.9_x.npy","./cnndata/cnn_train_0.9_y.npy")
    print("producting valid_0.1_x")
    mt(word2id,relation2id,"./data/valid_0.1.txt","./data/valid_0.1_x.npy","./cnndata/cnn_valid_0.1_y.npy")

def mt_seperate(path,path_word,path_pos1,path_pos2):
    x_train = np.load(path)
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for x in x_train:
        word = []
        pos1 = []
        pos2 = []
        for tmp in x:
            word.append(tmp[0])
            pos1.append(tmp[1])
            pos2.append(tmp[2])
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save(path_word, train_word)
    np.save(path_pos1, train_pos1)
    np.save(path_pos2, train_pos2)

def seperate():
    print("producing train_cnn")
    mt_seperate("./data/train_x.npy","./cnndata/cnn_train_word.npy","./cnndata/cnn_train_pos1.npy","./cnndata/cnn_train_pos2.npy")
    print("producing test_cnn")
    mt_seperate("./data/test_x.npy", "./cnndata/cnn_test_word.npy", "./cnndata/cnn_test_pos1.npy",
                "./cnndata/cnn_test_pos2.npy")
    print("producing train_0.9_cnn")
    mt_seperate("./data/train_0.9_x.npy", "./cnndata/cnn_train_0.9_word.npy", "./cnndata/cnn_train_0.9_pos1.npy",
                "./cnndata/cnn_train_0.9_pos2.npy")
    print("producing valid_0.1_cnn")
    mt_seperate("./data/valid_0.1_x.npy", "./cnndata/cnn_valid_0.1_word.npy", "./cnndata/cnn_valid_0.1_pos1.npy",
                "./cnndata/cnn_valid_0.1_pos2.npy")

def init_entityebd():
    dict_entityname2id = {}
    print('reading train data...')
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[2]
        en2 = content[3]
        if en1 not in dict_entityname2id.keys():
            dict_entityname2id[en1] = len(dict_entityname2id)
        if en2 not in dict_entityname2id.keys():
            dict_entityname2id[en2] = len(dict_entityname2id)

    with open('data/dict_entityname2id.pkl','wb') as output:
        pickle.dump(dict_entityname2id,output)

if __name__  == '__main__':
    print("producting dict_entityname2id")
    init_entityebd()
    print("producting bag_x")
    init_data()
    print("producting seperate x")
    seperate()
    print ('finished init!')