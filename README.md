# **Reinforced Natural Language Inference for Distantly Supervised Relation Classification**
## Data
To run this code, you need to download the following files([Baidu Yun](https://pan.baidu.com/s/11l9w8F4-FxUh-2ckEnxTrg) password: u031) in the `origin_data` folder.
 - relation2id.txt
 - relation2q.txt
 - train.txt
 - test.txt
 - vec.txt
 - ha_test1.txt
 - ha_test2.txt
 
 ## Initial
 In our experiment, we use [bert-as-service](https://github.com/hanxiao/bert-as-service) to encode sentences.
 Confirm that the `bert-as-service`has been deployed before the experiment, and modify the port in `initial_bert.py`
 Then you need to perform the following initialization operations in turn.
 
```
python initial_bert.py
```

```
python make_valid.py
```

```
python initial.py
```

```
python single_initial.py
```
## Train

```
python rlmodel.py
```

```
python select.py
```

```
python cnnmodel.py
```
## Test

```
python tp_sentence_test
```

 
