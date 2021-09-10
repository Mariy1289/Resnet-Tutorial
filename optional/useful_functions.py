import torch.nn as nn 
import matplotlib.pyplot as plt 
import datetime 
import math

# importing files 
# import sys
# import csv
# sys.path.append('../')
# from fn.set_fn import set_layer_fn
# from argument import parse_args

import os 
import csv
def save_to_csv(path):
    #write
    os.makedirs(path, exist_ok=True)
    with open('{}.csv'.format(path), 'w') as f:
        writer0 = csv.writer(f)
        writer0.writerow('weightlist')
        writer0.writerow('encodelist')
        writer0.writerow('fc')
    print("#### finish ####")
    
    #read
    with open('model/fn_param/{}.csv'.format(path)) as f:
        reader = csv.reader(f)
        weightlist = []
        for row in reader :
            weightlist.append(row)


#Conv2d内の重みを固定する関数
def freez_param (model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            m.weight.requires_grad = False

# 重みの中身を表示させる関数　std＊2は、大体95パーセントをカバーする
def get_std(mytensor):
    std_w = float(mytensor.mean().abs() + 2 * mytensor.std())
    fn = round(-math.log2(std_w))
    return fn


#注意：　getと言いつつも、意外と重い関数です。
# nn.Conv2dでself. として設定した任意のパラメータをとってくることができます。
def get_param(model):
    mylist = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            mylist.append(get_std(m.weight))
    print(mylist)


import datetime 
import matplotlib.pyplot as plt 
import os 
#get time and date 
def get_time():
    dt_now = datetime.datetime.now()
    date = dt_now.month + dt_now.day
    time = dt_now.hour+ dt_now.minute + dt_now.microsecond
    return date,time 

# plotting tensors ans save fig
def tensor_to_fig(input,output):
    plt.figure()
    plt.scatter(input.cpu().numpy(),output.cpu().numpy())
    date,time = get_time()
    new_dir_path_recursive = f'myfig/{date}'
    os.makedirs(new_dir_path_recursive, exist_ok=True)
    plt.savefig(f'{new_dir_path_recursive}/{time}')