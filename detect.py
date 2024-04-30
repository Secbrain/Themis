import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import *
from torch.autograd import Variable
import numpy as np
from PIL import Image
import argparse

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

model_name = ['resnet', 'inception', 'alexnet', 'vgg19', 'squeeze']
logo_name = ['AAAI', 'ACM', 'IEEE', 'Springer', 'USENIX', 'Berkely', 'Cambridge', 'CMU', 'MIT', 'SU', 'black', 'blue', 'gray', 'green', 'red']

parser = argparse.ArgumentParser()
parser.add_argument('--attack_model', type=str, default='squeeze')
parser.add_argument('--attack_logo', type=str, default='IEEE')
args = parser.parse_args()

model = torch.load('./model/' + args.attack_model + '_normal.pth')
model.eval()
model.cuda()
model_empty = torch.load('./model/' + args.attack_model + '_empty.pth')
model_empty.eval()
model_empty.cuda()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)
norm_ope = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# determine the threshold based on a set of clean samples 
threh = 7.783495623652154

# detect AEs
print('=== detect AEs ===')
for i1 in os.listdir('./aes/squeeze/'):
    img = Image.open('./aes/squeeze/' + i1).convert("RGB")
    img = transform(img)
    img = norm_ope(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    pred = model(img.cuda())
    predict = np.argmax(pred.cpu().detach().numpy())
    s1 = torch.softmax(pred, dim = 1)
    conf = s1[0][predict].item()
    pred_empty = model_empty(img.cuda())
    s1_empty = torch.softmax(pred_empty, dim = 1)
    conf_empty = s1_empty[0][predict].item()
    pvi = -np.log2(conf_empty) + np.log2(conf)
    if pvi < threh:
        print('Detection: it is an AE')
    else:
        print('Detection: it is not an AE')


# detect no-success (non-AEs)
print('=== detect no-success (non-AEs) ===')
for i1 in os.listdir('./non-aes/no_success/'):
    img = Image.open('./non-aes/no_success/' + i1).convert("RGB")
    img = transform(img)
    img = norm_ope(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    pred = model(img.cuda())
    predict = np.argmax(pred.cpu().detach().numpy())
    s1 = torch.softmax(pred, dim = 1)
    conf = s1[0][predict].item()
    pred_empty = model_empty(img.cuda())
    s1_empty = torch.softmax(pred_empty, dim = 1)
    conf_empty = s1_empty[0][predict].item()
    pvi = -np.log2(conf_empty) + np.log2(conf)
    if pvi < threh:
        print('Detection: it is an AE')
    else:
        print('Detection: it is not an AE')


# detect clean samples (non-AEs)
print('=== detect clean samples (non-AEs) ===')
for i1 in os.listdir('./non-aes/clean/'):
    img = Image.open('./non-aes/clean/' + i1).convert("RGB")
    img = transform(img)
    img = norm_ope(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    pred = model(img.cuda())
    predict = np.argmax(pred.cpu().detach().numpy())
    s1 = torch.softmax(pred, dim = 1)
    conf = s1[0][predict].item()
    pred_empty = model_empty(img.cuda())
    s1_empty = torch.softmax(pred_empty, dim = 1)
    conf_empty = s1_empty[0][predict].item()
    pvi = -np.log2(conf_empty) + np.log2(conf)
    if pvi < threh:
        print('Detection: it is an AE')
    else:
        print('Detection: it is not an AE')

