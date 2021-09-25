### 概要
cifar10画像用に修正されたDNNコードです。
torchvision のコードから引用
### 変更
- 最初のconv層のパラメータ変更
- ``self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)``

### 実行方法
```
python 3 main.py --model resnet18 --load True --save_weight True 
```
- model : resnet18,lenet,vgg,mlp_mixier 
- load : Trueの時は重みをロードして学習。Falseではスクラッチから学習
- save_weight :　True時に重みを保存。デバック時に軽くするために追加したオプション。
- lr : learning rate . inferenceTrueの際には、lrを下げた方が早い。


**その他**
- 重みのパスは``model_choice ():``で定義する
- 最高精度：　92.740%



**models**
```
$ pip install mlp-mixer-pytorch
```
### MLP_mixier 
model code from https://github.com/lucidrains/mlp-mixer-pytorch

- ResMLP
    - [paper](https://arxiv.org/pdf/2105.03404.pdf)
    - [original code](https://github.com/rishikksh20/ResMLP-pytorch)
    - inspired by the vision transformers (ViT)
    - replace the self-attention sublayer by a linear layer
    - remove the need for batch-specific or cross-channel normalizations such as BatchNorm, GroupNorm or LayerNorm.
    - use gelu as activation layer 
- sMLP
- MLPmixier 