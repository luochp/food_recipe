#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import transforms
import glob
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torch import optim
import torch
import numpy as np
import time
from torch import optim, cuda
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
import sys

print('Start')

#if len(sys.argv) != 2:
 #   print("Wrong parameter")
  #  sys.exit(-1)

# check cuda environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 100

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root="../recipe-box/data_for_nn/train/", transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root="../recipe-box/data_for_nn/valid/", transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root="../recipe-box/data_for_nn/test/", transform=image_transforms['valid']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

# Iterate through the dataloader once
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
features.shape, labels.shape


# In[2]:


def get_pretrained_model(model_name, n_classes = 100, device = device):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """
    model = None
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, n_classes), 
            nn.LogSoftmax(dim=1))
        
        model.classifier[3].weight.requires_grad = True
        model.classifier[3].bias.requires_grad = True

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, n_classes), 
            nn.LogSoftmax(dim=1))

    # Move to gpu and parallelize
    if device.type == 'cuda':
        model = model.to('cuda')
        model = nn.DataParallel(model)
    
    if cuda.device_count()>1:
        summary(
            model.module,
            input_size=(3, 224, 224),
            batch_size=batch_size,
            device='cuda')
    else:
        summary(
            model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return model


# In[3]:


model = get_pretrained_model('vgg16')


# In[12]:


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print("Model has been trained for: %d epochs.\n" %(model.epochs))
    except:
        model.epochs = 0
        print("Starting Training from Scratch.\n")

    overall_start =  time.time()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = time.time()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            
            # Calculate Top K accuracy
            maxk = 5
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct_tensor = pred.eq(target.view(1, -1).expand_as(pred))
            train_acc += correct_tensor[:maxk].view(-1).float().sum(0)
            
            # Track training progress
            print("Epoch: %d \t %.2f%% complete. %.2f seconds elapsed in epoch."% (epoch, 100 * (ii + 1) / len(train_loader), time.time() - start))

        # After training loops ends, start validation
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if device.type == 'cuda':
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                output = model(data)

                # Validation loss
                loss = criterion(output, target)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate Top K accuracy
                maxk = 5
                batch_size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct_tensor = pred.eq(target.view(1, -1).expand_as(pred))
                valid_acc += correct_tensor[:maxk].view(-1).float().sum(0)

            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(valid_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print("\nEpoch: %d \tTraining Loss: %.4f \tValidation Loss: %.4f" % (epoch, train_loss, valid_loss))
                print("\t\tTraining Accuracy: %.2f \t Validation Accuracy: %.2f" % (100 * train_acc, 100 * valid_acc))

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print("\nEarly Stopping! Total epochs: %d. Best epoch: %d with loss: %.2f and acc: %.2f" % (epoch, best_epoch, valid_loss_min, 100 * valid_acc))
                    total_time =  time.time() - overall_start
                    print("%.2f total seconds elapsed. %.2f seconds per epoch." % (total_time, total_time / (epoch+1)))

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc'
                        ])
                    return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time =  time.time() - overall_start
    print("\nBest epoch: %d with loss: %.2f and acc: %2f" %(best_epoch, valid_loss_min, 100 * valid_acc))
    print("%.2f total seconds elapsed. %.2f seconds per epoch." % (total_time, total_time / (epoch)))
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


# In[13]:


#save_file_name = 'model_state_%s.pt' % (sys.argv[1])
save_file_name = "model_state.pt"
# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['valid'],
    save_file_name=save_file_name,
    max_epochs_stop=12,
    n_epochs=2,
    print_every=1)


# In[21]:


from sklearn.externals import joblib
# save category classes
#joblib.dump((model, history), './last_model_and_history_%s.pkl' % (sys.argv[1]))
joblib.dump((model, history), "./last_model_and_history.pkl")


print('Done')

