# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:31
# @Author  : zhoujun

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, lmdb_save_path, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        data_list  : a list contains img_path\tlabel
        lmdb_save_path : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(lmdb_save_path, exist_ok=True)
    env = lmdb.open(lmdb_save_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc=f'make dataset, save to {lmdb_save_path}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)



def createDatasetv2(data_list, lmdb_save_path, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        data_list  : a list contains img_path\tlabel
        lmdb_save_path : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(lmdb_save_path, exist_ok=True)
    env = lmdb.open(lmdb_save_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    #writer = jsonlines.open(jsonl_save_path, 'w')

    for imagePath, label in tqdm(data_list, desc=f'make dataset, save to {lmdb_save_path}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        #writer.write({'img_id':f'image_{cnt:09d}','label':label,'lmdb_name':lmdb_name})
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def run():

    label_file = r"/data/wufan/data/database/txt/val.txt"
    lmdb_save_path = r'/data/wufan/data/database/lmdb/synthetic_chinese_dataset/val'
    os.makedirs(lmdb_save_path, exist_ok=True)

    data_list = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'load data from {label_file}'):
            line = line.strip('\n').replace('.jpg ', '.jpg\t').replace('.png ', '.png\t').split('\t')
            if len(line) > 1:
                img_path = pathlib.Path(line[0].strip(' '))
                label = line[1]
                if img_path.exists() and img_path.stat().st_size > 0:
                    data_list.append((str(img_path), label))

    createDataset(data_list, lmdb_save_path)


if __name__ == '__main__':
    import pathlib
    import json
    import jsonlines
    mode = 'test'
    # label_path = '/data/wufan/text_render/example_data/output/word_corpus/labels.json'
    label_path = f'/data/wufan/text_render/example_data/output/char_corpus/label_{mode}_mini.json'
    # img_dir = '/data/wufan/text_render/example_data/output/word_corpus/images'
    img_dir = '/data/wufan/text_render/example_data/output/char_corpus/images'
    lmdb_save_path = f'/data/wufan/data/lmdb_data/zh_4m_mini/{mode}'
    lmdb_name = os.path.basename(lmdb_save_path)
    # jsonl_save_path = f'/data/wufan/data/database/jsonl/{lmdb_name}.jsonl'

    data_list = []
    with open(label_path,encoding='utf-8') as rfile:
        data = json.load(rfile)
        labels = data['labels']
        pbar = tqdm(labels)
        for name in pbar:
            img_path = os.path.join(img_dir,name+'.jpg')
            data_list.append((img_path,labels[name]))

    createDatasetv2(data_list, lmdb_save_path)


