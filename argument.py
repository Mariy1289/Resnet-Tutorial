import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--epochs', type=int, default=70)
    parser.add_argument( '--lr', type=float, default=0.0005)
    parser.add_argument('-ss', '--step_size', type=int, default=30)
    parser.add_argument('--model', type=str, default='resnet18', choices=['lenet','resnet18'])
    parser.add_argument('--load', type=str, default=False)
    parser.add_argument('--save_weight', type=str, default=False)
    return parser.parse_args()