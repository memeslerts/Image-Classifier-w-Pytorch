import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models
from collections import OrderedDict
from torch import nn
from torch import optim
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', help = 'Select dataset directory', type = str)
parser.add_argument('--save_dir', help = 'Select directory for saving', type = str, default = './')
parser.add_argument('--arch', help = 'Choose architecture', type = str, default = 'vgg16')
parser.add_argument('--epochs',help = 'Number of epochs',type = int,default = 5)
parser.add_argument('--hiddenlayers',help = 'Unit for hidden layers',type = int,default = 256)
parser.add_argument('--learnrate',help='Unit for learning rate',type = float, default = 0.001)
parser.add_argument('--dropout',help='Unit for dropouts',type = int,default = 0.2)
parser.add_argument('--gpu',help='Work in a GPU environment',type = str, default = "cpu")
parser.add_argument('--category_names',help='Converting category names',type = str, default = 'cat_to_name.json')
    
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


if data_dir:

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    valid_transforms= transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(valid_dir,transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

def load_model(in_feat,dropout,hiddenlayers,out_feat,category_names,arch,gpu,epochs,learnrate):
    with open(args.category_names,'r') as f:
        cat_to_name = json.load(f)
                    
    if arch == 'vgg16':
        model = model.vgg16(pretrained = True)
        in_feat = 25088
    elif arch == 'alexnet':
        model = model.alexnet(pretrained = True)
        in_feat = 9216
    out_feat = len(cat_to_name)               

    for param in model.parameters():
        param.requires_grad = False  
        
        
    if args.gpu == "gpu":
        device = 'cuda'
    else:
        device = 'cpu"
#build a feedforward classifier    
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(in_feat,hiddenlayers)),
                                        ('relu1',nn.ReLU()),
                                        ('fc2',nn.Linear(hiddenlayers,hiddenlayers)),
                                        ('relu2',nn.ReLU()),
                                        ('dropout',nn.Dropout(dropout))
                                        ('fc3',nn.Linear(hiddenlayers,out_feat)),
                                        ('output',nn.LogSoftmax(dim=1))]))
    model.to(device)                                        
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learnrate)
                                            
    model.to(device);

    epochs = args.epochs

    steps = 0

    print_every = 30

    for e in range(epochs):
    
        running_loss = 0
    
        for images, labels in trainloader:
                
            steps+=1
        
            images, labels = images.to(device),labels.to(device)
                
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps % print_every == 0:
        
                model.eval()
            
                with torch.no_grad():
                
                    test_loss,accuracy = Validation(model,validloader)
                
                print("Epoch: {}/{}..".format(e+1,epochs),
                 "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
                 "Testing Loss: {:.3f}..".format(test_loss/len(testloader)),
                 "Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
                running_loss = 0
            
                model.train()
            
                                            
                                            
                                            
   

    
def Validation(model,testloader):
    test_loss = 0
    accuracy = 0
    for images, labels in iter(testloader):
            
        images, labels = images.to(device),labels.to(device)
            
        log_ps = model(images)
            
        b_loss = criterion(log_ps, labels)
        
        test_loss += b_loss.item()
            
        ps = torch.exp(log_ps)
            
        top_p, top_class = ps.topk(1,dim=1)
            
        equals = top_class == labels.view(*top_class.shape)
            
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
    return test_loss, accuracy

def checkpoint(arch,classifier,epoch,optimizer,state_dict,mapping,save_dir):
    checkpoint = {'arch': model.arch,
             'classifier': model.classifier,
             'epoch':model.epochs,
             'optimizer':model.optimizer.state_dict(),
             'state_dict':model.state_dict(),
             'mapping': model.class_to_idx}
             
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
                                            
load_model(args.arch,arch.category_names,args.epochs,args.hiddenlayers,args.learnrate,args.dropout,args.gpu,args.save_dir)