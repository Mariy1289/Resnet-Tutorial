import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os 

from argument import parse_args
import torch.optim as optim
import sys


def get_optimizer(model, optimizer, lr,momentum_):
    if optimizer == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr,
            momentum = momentum_)
            # weight_decay=1e-4)
            # ,weight_decay=5e-4)
        print("optimizer is SGD")
    elif optimizer == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr)
        print("optimizer is Adam")
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr)
        print("optimizer is AdamW")
    elif optimizer == 'rprop':
        optimizer = optim.Rprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr)
        print("optimizer is Rprop")
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr)
        print("optimizer is RMSprop")
    else:
        print("#### unknown optimizer ####")
        sys.exit()
    return optimizer



def save_weight(model,basename):
    args = parse_args()
    # file_name = '%s_%s%s(%s%s)'%(args.model,args.encode,args.weight,args.M,args.K)
    # save_weight_path = 'weights/%s_e%sw%s/(%s%s)'%(args.model,args.encode,args.weightquant,args.M,args.K)
    save_weight_path = 'weights/%s'%(basename)
    os.makedirs(save_weight_path, exist_ok=True)
    save_args_as_json(args, save_weight_path)
    torch.save(model.state_dict(),
                '%s/_best.pth' % (save_weight_path))
    return save_weight_path

def save_args_as_json(args, save_weight_path):
    args_list = [(i, getattr(args, i)) for i in dir(args) if not '_' in i[0]]
    with open('%s/argparse_setting.json' % save_weight_path, 'w') as f:
        for i, j in args_list:
            json.dump({i: j}, f)
    print("#### save argparse as json file ####")



def accuracy(loader, model, device=None):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            if device is None:
                images, labels = data
            else:
                images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def accuracy_batch(outputs, labels):
    total = 0
    correct = 0
    total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return 100.0 * correct / total


def accuracy_of_classes(num_classes, loader, model, device=None):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in loader:
            if device is None:
                images, labels = data
            else:
                images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return [100.0 * correct / total for correct, total in zip(class_correct, class_total)], (100.0 * sum(class_correct) / sum(class_total))



def images_to_probs(net, images, output):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
#     output = net(images.to(device))
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, outputs, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images, outputs)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def show_table(data):
    # https://stackoverflow.com/questions/35160256/how-do-i-output-lists-as-a-table-in-jupyter-notebook
    display(HTML(tabulate.tabulate(data, tablefmt='html')))


def add_param(writer, net, gstep):
    
    for name, value in net.state_dict().items():
        # breakpoint()
        writer.add_histogram(name, value, gstep)



class IntermediateOutputWriter(object):
    def __init__(self, writer, net, step):
        super(IntermediateOutputWriter, self).__init__()
        self.writer = writer
        self.net = net
        self.hooks = []
        self.step = step
    
    def __enter__(self):
        class _f:
            def __init__(self, writer, name, step):
                self.wrote = False
                self.name = name
                self.step = step
                self.writer = writer
            def __call__(self, m, i, o):
                if not self.wrote:
                    self.writer.add_histogram(self.name, o, self.step)
                    self.wrote = True
        for name, module in self.net.named_modules():
#             print("Register: %s" % name)
            self.hooks.append(module.register_forward_hook(
                _f(self.writer, "%s.output" % name, self.step)
            ))
        return self

    def __exit__(self, ex_type, ex_value, trace):
        while len(self.hooks) > 0:
            self.hooks.pop().remove()
        return False

