#   Example call:
#    python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k 3
##

# Imports python modules
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image

# Main program function defined below
def main():    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # prints command line agrs
    print("Command Line Arguments:\n     image path =", in_arg.image_path,
         "\n    save dir = ", in_arg.Checkpoint,
          "\n    top k = ", in_arg.top_k,
         "\n    gpu = ", in_arg.gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    
    model = load_and_rebuild_model(in_arg.Checkpoint)        
    model.to(device)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    probs, labels = predict(in_arg.image_path, model, cat_to_name, device, in_arg.top_k)
    for x, y in zip(probs, labels):
        print(y, ':', x)

# Functions defined below
def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str,
                        help='path of image')
    parser.add_argument('Checkpoint', type=str, default='Checkpoint.pth', 
                        help='path of Checkpoint')
    parser.add_argument('--gpu', action='store_true',
                        help='using gpu')
    parser.add_argument('--top_k', type=int, default='5', 
                        help='top k')
    
    # returns parsed argument collection
    return parser.parse_args()    

def load_and_rebuild_model(filepath):
    cp = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    if cp['arch'] == 'vgg':
        model = models.vgg16(pretrained=False)
        model_input = 25088
    elif cp['arch'] == 'alexnet':    
        model = models.alexnet(pretrained=False)   
        model_input = 9216
    else:
        print("Sorry base architecture note recognized")
        exit(1)     

    for param in model.parameters():
        param.requires_grad = False     
    h_input = cp['hidden_units']
    classifier = nn.Sequential(OrderedDict([
                              ('fc0', nn.Linear(model_input, h_input)),
                              ('relu0', nn.ReLU()),
                              ('dropout0', nn.Dropout(p=0.25)),    
                              ('fc1', nn.Linear(h_input, 512)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.25)),
                              ('fc2', nn.Linear(512, 102)),
                              ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier        
    model.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['model_state'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # load the image
    img_pil = Image.open(image)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    
    # preprocess the image
    img_tensor = preprocess(img_pil)

    return img_tensor

def predict(image_path, model, cat_to_name, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img_tensor = process_image(image_path)
    model_input = img_tensor.unsqueeze(0)
    model_input = model_input.to(device)
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(model_input)
        
    ps = torch.exp(output)
    
    ps_topk, ps_topk_indices = ps.topk(topk)
    ps_topk = ps_topk.data.cpu().numpy()[0]
    ps_topk_indices = ps_topk_indices.data.cpu().numpy()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}    
    topk_labels = [int(idx_to_class[lab]) for lab in ps_topk_indices]
    
    topk_flowers = [idx_to_class.get(key) for key in topk_labels]
    topk_flowers = [cat_to_name.get(key) for key in topk_flowers]
    
    return ps_topk, topk_flowers

# Call to main function to run the program
if __name__ == "__main__":
    main()
