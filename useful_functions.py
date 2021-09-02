import torch.nn as nn 
import matplotlib.pyplot as plt 
import datetime 
import math

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