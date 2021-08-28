#Conv2d内の重みを固定する関数
import torch.nn as nn 

def freez_param (model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            m.weight.requires_grad = False


# 重みの中身を表示させる関数　std＊2は、大体95パーセントをカバーする
def get_param(model):
    mylist = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            mylist.append(m.weight.mean()+2*m.weight.std())
    print(mylist)



