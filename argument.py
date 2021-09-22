import argparse
import torch
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--epochs', type=int, default=70)
    parser.add_argument( '--lr', type=float, default=0.01)
    parser.add_argument( '--M', type=int, default=2)
    parser.add_argument( '--K', type=int, default=2)
    parser.add_argument( '--encode',type=str, default=False)
    parser.add_argument( '--weightquant','--wk',type=str, default=False)
    parser.add_argument('-ss', '--step_size', type=int, default=30)
    parser.add_argument('--model', type=str, default='resnet18', choices=['lenet','resnet18','vgg11','mixier'])
    parser.add_argument('-opt', '--optimizer',type=str,default='sgd',
                        choices=['sgd', 'adam', 'adamw', 'rprop', 'rmsprop'])
    parser.add_argument('--load', type=str, default=False)
    parser.add_argument('--save_weight', type=str, default=False)
    parser.add_argument('--auto_fn', type=str, default=False)
    parser.add_argument('--std', type=int, default=3)
    parser.add_argument('--encode_later', type=str, default=False)
    parser.add_argument('-d', '--device', type=int or str,default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('-qm','--quantize_method', type=str, default='bwb', choices=['bwb','eq15'])
    parser.add_argument('--csv', type=str, default=True)
    parser.add_argument('-g', '--gamma', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.9)

    return parser.parse_args()
