##   Example call:
#    python train.py flowers --save_dir checkpoint.pth --arch vgg --learning_rate 0.01 --hidden_units 1024 --epochs 10 --gpu
#
#    python train.py flowers --save_dir checkpoint.pth --arch alexnet --learning_rate 0.01 --hidden_units 1024 --epochs 10 --gpu
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

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'alexnet': alexnet, 'vgg': vgg16}

# Main program function defined below
def main():    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # prints command line agrs
    print("Command Line Arguments:\n     dir =", in_arg.folder,
         "\n    save_dir = ", in_arg.save_dir,
         "\n    arch = ", in_arg.arch,
         "\n    learning_rate = ", in_arg.learning_rate,
         "\n    hidden_units = ", in_arg.hidden_units,
         "\n    epochs = ", in_arg.epochs,
         "\n    gpu = ", in_arg.gpu)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    
    train_dir = in_arg.folder + '/train'
    valid_dir = in_arg.folder + '/valid'
    test_dir = in_arg.folder + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.CenterCrop(10),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # TODO: Build and train your network
    model = models[in_arg.arch]
    
    for param in model.parameters():
        param.requires_grad = False
        
    model_input = 9216 if in_arg.arch == 'alexnet' else 25088
    h_input = in_arg.hidden_units
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

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=in_arg.learning_rate, momentum=0.5)
    
    do_deep_learning(model, trainloader, validloader, in_arg.epochs, 40, criterion, optimizer, device)
    check_accuracy_on_test(model, testloader, device)
    
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': in_arg.arch,
                  'hidden_units': h_input,
                  'model_state': model.state_dict(),
                  'criterion_state': criterion.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, in_arg.save_dir)    
    
# Functions defined below
def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str, default='flowers/', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                        help='path to checkpoint')
    parser.add_argument('--arch', type=str, default='vgg', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default='0.005', 
                        help='chosen model')
    parser.add_argument('--hidden_units', type=int, default='1024', 
                        help='chosen model')
    parser.add_argument('--epochs', type=int, default='10', 
                        help='chosen model')
    parser.add_argument('--gpu', action='store_true',
                        help='using gpu')
    
    # returns parsed argument collection
    return parser.parse_args()    

# Implement a function for the validation pass
def validation(model, validloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    model.train()
    epochs = epochs
    print_every = print_every
    steps = 0
    
    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()

# TODO: Do validation on the test set
def check_accuracy_on_test(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on test images: %d %%" % (100 * correct / total))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
