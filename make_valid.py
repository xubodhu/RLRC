import random
import numpy as np
import pickle
print('reading origin data...')
f = open('./origin_data/train.txt', 'r', encoding='utf-8')
cls_vec = np.load("./data/cls.npy")


sen = {}
i=0
num = 0
while True:
    content = f.readline()
    if content == '':
        break
    tmp = content.strip().split("\t")
    if(tmp[0]+tmp[1] in sen.keys()):
        sen[tmp[0]+tmp[1]].append((content,cls_vec[i],i))
    else:
        num += 1
        print(num)
        sen[tmp[0]+tmp[1]]=[]
        sen[tmp[0]+tmp[1]].append((content,cls_vec[i],i))
    i+=1


print("making valid_data_0.1 and train_data_0.9 and cls_vec_0.9")
valid_data = []
train_data = []
cls_vec_09 = []
train_index = []
for key in sen.keys():
    flag = random.random()
    if(flag <= 0.1):
        for tmp in sen[key]:
            valid_data.append(tmp[0])
    else:
        for tmp in sen[key]:
            train_data.append(tmp[0])
            cls_vec_09.append(tmp[1])
            train_index.append(tmp[2])

print(len(train_data))
print(len(valid_data))
print("saving data")
with open("./data/train_0.9.txt","w",encoding="utf-8") as f:
    for line in train_data:
        f.write(line)


with open("./data/valid_0.1.txt","w",encoding="utf-8") as f:
    for line in valid_data:
        f.write(line)


cls_vec_09 = np.array(cls_vec_09)
np.save("./data/cls_vec_0.9",cls_vec_09)

with open("./data/train_index.txt","w",encoding="utf-8") as f:
    f.write(str(train_index))

#得到label_entity_pair
def produce_label_data():

    with open('./data/valid_0.1.txt','r',encoding= 'utf-8') as input:
        test_data = input.readlines()
    print(len(test_data))
    dict_relation2id = {}
    label_entitypair = {}


    with open('origin_data/relation2id.txt','r',encoding='utf-8') as input:
        lines = input.readlines()

    for line in lines:
        line = line.strip()
        relation = line.split()[0]
        id = int (line.split()[1])
        dict_relation2id[relation] = id

    for line in test_data:
        line = line.strip()
        items = line.split('\t')
        e1 = items[0]
        e2 = items[1]
        if(items[4] in dict_relation2id.keys()):
            relationid =dict_relation2id[items[4]]
        else:
            relationid=0
        key = e1+'$'+e2
        if key not in label_entitypair.keys():
            label_entitypair[key] = set()
            label_entitypair[key].add(relationid)
        else:
            label_entitypair[key].add(relationid)

    num_entitypair = len(label_entitypair)
    num_entitypair_true = 0
    for key in label_entitypair.keys():
        tmp_set = label_entitypair[key]
        if len(tmp_set) >1:
            num_entitypair_true+=1
        elif 0 not in tmp_set:
            num_entitypair_true += 1

    print ('num_entitypair：',num_entitypair)
    print ('num_entitypair_true:',num_entitypair_true)

    with open('data/valid_0.1_label_entitypair.pkl','wb') as output:
        pickle.dump(label_entitypair,output)
    return (num_entitypair_true)

print("producing valid_label_entitypair")
num = produce_label_data()
f = open("./data/num_entitypair_true","w")
f.write(str(num))



