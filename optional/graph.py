import matplotlib.pyplot as plt
import numpy as np
import datetime
# x軸の範囲を指定
left= 0 
right = 8

def addlabels(x,y):
    for i in range(len(x)):
        if i ==0  or i ==len(x)-1:
            plt.text(x[i], y[i], y[i], ha = 'center')


def make_fig(x,y,bench,model):
    left= 0
    right = 8
    if model =='resnet18': c = 'g'
    elif model =='vgg11':c = 'b'
    else :c ='k'
    plt.hlines(bench, left,right, c, linestyles='dashed',label="baseline: {} ".format(model))
    plt.plot(x,y,c,marker='o',label = 'Quantized, Cifar10,{}'.format(model))
    # plt.title("dataset: Cifar10, model: Resnet")
    addlabels(x, y)
    
    


def get_time():
    dt_now = datetime.datetime.now()
    date = dt_now.month + dt_now.day
    time = dt_now.hour+ dt_now.minute + dt_now.microsecond
    return date,time 



x = np.array([1,  2,        3,      5,       8])
y = np.array([70.7,90.92,91.590,   92.200,      91.860,    ])
y = y/100
y[0] = 0.707
bench =  0.92
model = 'resnet18'
make_fig(x,y,bench,model)



x = np.array([1,    2,  3,6,  8])
y = np.array([0.63770,0.8850,0.90120,0.91480,0.91390])
bench = 0.90420
model = 'vgg11'
make_fig(x,y,bench,model)



x = np.array([1,  2,  3,        4,          5,       6,         7,      8])
y = np.array([0.1,0.1,0.5575,   0.69260,    0.72750, 0.74460,   0.74970,     0.7522])
bench =  0.74920
model = 'lenet'
make_fig(x,y,bench,model)

plt.xlabel('weight,activation bit width')
plt.ylabel("test accuracy")
plt.legend(loc='lower right', borderaxespad=0, fontsize=8)
plt.xlim(1,right)
plt.ylim(0, 1)
date,time = get_time()
plt.savefig('{}.png'.format(date,model))



