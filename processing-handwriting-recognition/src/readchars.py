# coding: utf-8
import csv
import json
import os
import csv
import os.path
import numpy as np
from scipy.misc import imread

def read_position_data(num):
    dict_pos = dict()
    for line in open("Tdata/position_data"+str(num)+".txt"):
        l = line.strip().split("|")
        l[0] = int(l[0])
        pos = l[1].strip().split("	")
        for i in range(0,len(pos),1):
            pos[i] = float(pos[i])
            print type(pos[0]),pos[0]
        dict_pos[l[0]] = pos
    with open("Jdata/position_data"+str(num)+".json", 'w') as f:
        f.write(json.dumps(dict_pos, ensure_ascii=False, encoding='UTF-8', indent=4))
    f.close()

def read_triantest_sequence():
    train_list = []
    train_set = dict()
    test_set = dict()
    test_list = []
    for line in open("Tdata/unigram_train_label.txt"):
        l = line.strip().split("|")
        train_list.append(str(l[0]))
        label = l[1].strip().split("	")
        train_set[str(l[0])] = label
    for line in open("Tdata/unigram_test_label.txt"):
        l = line.strip().split("|")
        test_list.append(str(l[0]))
        label = l[1].strip().split("	")
        test_set[str(l[0])] = label

    with open("Jdata/new_train_sequence.json", 'w') as f:
        f.write(json.dumps(train_list, ensure_ascii=False, encoding='UTF-8', indent=4))
    with open("Jdata/new_test_sequence.json", 'w') as f:
        f.write(json.dumps(test_list, ensure_ascii=False, encoding='UTF-8', indent=4))
    with open("Jdata/uni_train_set.json", 'w') as f:
        f.write(json.dumps(train_set, ensure_ascii=False, encoding='UTF-8', indent=4))
    with open("Jdata/uni_test_set.json", 'w') as f:
        f.write(json.dumps(test_set, ensure_ascii=False, encoding='UTF-8', indent=4))


#读取part型训练数据集
def read_part_train_image(flag,addnum,dir):
    rootdir = dir  # 指明被遍历的文件夹
    train_sequence = json.load(open("Jdata/uni_sequence.json"))
    train_set = json.load(open("Jdata/uni_all_label.json"))
    position = json.load(open("Jdata/position_data"+str(addnum)+".json"))

    im_train = []
    position_train = []
    i = 0

    for relname in train_sequence:
        image_dir = rootdir + "/" + relname+".png"
        im = imread(image_dir)
        # print im.shape
        # print im
        im.shape = -1
        # print im
        # print im.shape
        for l in range(0,len(train_set[relname]),1):
            im_train.append(im)
            position_train.append(np.array(position[str(l)],dtype='float16'))
        i += 1
        if i == flag:
            break

    mat_train = np.array(im_train, dtype='int16')
    mat_position_train = np.array(position_train, dtype='float16')

    print mat_train.shape,mat_position_train.shape
    return [mat_train,mat_position_train]

    np.savetxt('data/part/image_train_data.csv', mat_train, delimiter=',')
    np.savetxt('data/part/image_train_addData.csv', mat_position_train, delimiter=',')
#读取测试数据集
def read_test_image(flag,addnum,dir):
    rootdir = dir  # 指明被遍历的文件夹
    test_sequence = json.load(open("Jdata/word_test_list.json"))
    #test_set = json.load(open("Jdata/test_set.json"))
    position = json.load(open("Jdata/position_data"+str(addnum)+".json"))

    im_test = []
    position_test = []
    j = 0

    for relname in test_sequence:
        image_dir = rootdir + "/" + relname + ".png"
        im = imread(image_dir)
        im.shape = -1
        for m in range(0, 15, 1):
            im_test.append(im)
            position_test.append(np.array(position[str(m)], dtype='float16'))
        j += 1
        if j == flag:
            break

    mat_test = np.array(im_test, dtype='int16')
    mat_position_test = np.array(position_test, dtype='float16')

    print mat_test.shape,mat_position_test.shape
    return [mat_test,mat_position_test]

    np.savetxt('data/part/image_test_data.csv', mat_test, delimiter=',')
    np.savetxt('data/part/image_test_addData.csv', mat_position_test, delimiter=',')
#读取part型训练类标
def get_part_train_label(flag):
    train_sequence = json.load(open("Jdata/uni_sequence.json"))
    train_set = json.load(open("Jdata/uni_all_label.json"))
    label_train = []
    i = 0
    for relname in train_sequence:
        label_train.extend(train_set[relname])
        i+=1
        if i==flag:
            break
    mat_label_train = np.array(label_train, dtype='int16')
    print mat_label_train.shape
    return  mat_label_train
    np.savetxt('data/part/label_train.csv', mat_label_train, delimiter=',')
#读取测试类标
def get_test_label(flag):
    test_sequence = json.load(open("Jdata/word_test_list.json"))
    test_set = json.load(open("Jdata/uni_test_set.json"))
    label_test = []
    j = 0
    for relname in test_sequence:
        lable_list = test_set[relname]
        rest_num = 15-len(lable_list)
        for i in range(0,rest_num,1):
            lable_list.append(str(59))
        label_test.extend(lable_list)
        j+=1
        if j==flag:
            break
    mat_label_test = np.array(label_test, dtype='int16')
    print mat_label_test.shape
    return mat_label_test
    np.savetxt('data/part/label_test.csv', mat_label_test, delimiter=',')

def process_result():
    dataSet = np.loadtxt(open("result.csv"),
                         delimiter=",")
    print dataSet.shape
    result_label = np.argmax(dataSet, axis=1)
    print (type(np.argmax(dataSet, axis=1)),np.argmax(dataSet, axis=1).shape)
    np.savetxt('result_label.csv', result_label, delimiter=',')
if __name__ == '__main__':
    pass
    # alist = [300]
    # for key in alist:
    #     read_position_data(key)
    #read_part_train_image(1, 200)
    # read_part_train_image()
    # read_test_image()
    read_triantest_sequence()
    # get_part_train_label()
    # get_test_label()
    #process_result()