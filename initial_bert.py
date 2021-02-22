from bert_serving.client import BertClient
import numpy as np

print("ready to make question")

r2q = {}
with open("./origin_data/relation2q.txt","r",encoding="utf-8") as f:
    while True:
        content = f.readline()
        if(content == ""):
            break
        else:
            content = content.strip().split("\t")
            r2q[content[0]] = content[1]

print("ready to make q_a_t")
q_a_t = []
with open("./origin_data/train.txt","r",encoding="utf-8") as f:
    while True:
        content = f.readline()
        if(content == ""):
            break
        else:
            content = content.strip().split("\t")
            e1 = content[2]
            e2 = content[3]
            relation = content[4]
            if(relation not in r2q.keys()):
                relation = "NA"
            text = content[5:-1]
            tmp_q = r2q[relation]
            tmp_q = tmp_q.replace("1",e1).replace("2",e2)
            tmp_q_a_t = text[0] + " ||| "+tmp_q
            q_a_t.append(tmp_q_a_t)

print("ready to make cls")
cls_list = []
bc = BertClient(port=3333,port_out=3334)
steps = len(q_a_t)//128
for i in range(steps):
    buf = q_a_t[i*128:(i+1)*128]
    vec = bc.encode(buf)
    print(vec.shape)
    for j in range(vec.shape[0]):
        cls_list.append(vec[j])
    print(str(i)+"/"+str(steps))
buf = q_a_t[(i+1)*128:]
vec = bc.encode(buf)
for j in range(vec.shape[0]):
    cls_list.append(vec[j])

print(len(cls_list))
np.save("./data/cls.npy",cls_list)


