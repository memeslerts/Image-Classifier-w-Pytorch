import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models
from collections import OrderedDict
from torch import nn
from torch import optim
from PIL import Image
import torch.nn.functional as F

parser = argparse.ArgumentParser()
#make a bunch of argument parsers
#pathtoimage
parser.add_argument('--path_to_image', help='Provide path to image',type = str, default = './flowers/test/1/image_06743.jpg')
#topk amount
parser.add_argument('--topk', help='Number of top k highest probabilities', type = int, default = 5)
#checkpoint
parser.add_argument('--checkpoint', help='Load checkpoint', type = str, default = './checkpoint.pth')
#categorynames
parser.add_argument('--category_names', help='Converting category names into strings', type = str, default = 'cat_to_name.json')
#gpu
parser.add_argument('--gpu', help = 'Work in a GPU environment', type = str, default = 'cpu')

args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] = 'vgg16':
        model.arch = model.vgg16(pretrained = True)
    elif checkpoint['arch'] = 'alexnet':
        model.arch = model.alexnet(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.epoch = checkpoint['epoch']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    return model

def process_image(image):

    im = Image.open(image)
    
    img_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    im = img_transform(im)
    
    np_image = np.array(im)
    
    return np_image

def predict(args.image_path, args.model, args.topk, args.device):

    model.eval()
    model.to(device)
    
    np_img = process_image(image_path)
    img = torch.from_numpy(np_img)
    img = img.unsqueeze_(0)
    img = img.float()
    img = img.to(device)
    
    with torch.no_grad():
        output = model.forward(img)

    prob = F.softmax(output.data, dim=1)

    return prob.topk(topk)

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
if args.category_names:
    with open(args.category_names,'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
        
probabilities = np.array(probabilities[0][0])

classes = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]

print("The highest probability classes are: {}".format(classes),
     "The probabilities are: {}".format(probabilities))
            
    
    