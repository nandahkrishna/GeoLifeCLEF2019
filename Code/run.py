import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from environmental_raster_glc import PatchExtractor

class GeoLifeClefDataset(Dataset):
    def __init__(self, extractor, dataset, labels):
        self.extractor = extractor
        self.labels = labels
        self.dataset = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor = self.extractor[self.dataset[idx]]
        return torch.from_numpy(tensor).float(), self.labels[idx]

if __name__ == '__main__':
    patch_extractor = PatchExtractor('../rasters GLC19', size=64, verbose=True)

    # patch_extractor.add_all()
    patch_extractor.append('chbio_1')
    patch_extractor.append('chbio_2')
    patch_extractor.append('chbio_3')
    patch_extractor.append('chbio_4')
    patch_extractor.append('chbio_5')
    patch_extractor.append('chbio_6')
    patch_extractor.append('chbio_7')
    patch_extractor.append('chbio_8')
    patch_extractor.append('chbio_9')
    patch_extractor.append('chbio_10')
    patch_extractor.append('text')

    # dataset
    df = pd.read_csv("../PL_trusted.csv",sep=';')
    df = pd.concat([df.drop('glc19SpId',axis=1),pd.get_dummies(df['glc19SpId'])], axis=1)

    dataset_list = list(zip(df["Latitude"],df["Longitude"]))
    labels_list = np.asarray(df.iloc[:, 10:])
    
    train_ds = GeoLifeClefDataset(patch_extractor, dataset_list[:230000], labels_list[:230000])
    test_ds = GeoLifeClefDataset(patch_extractor, dataset_list[230000:], labels_list[230000:])
    
    datasets = {"train": train_ds, "val": test_ds}
    
    trainloader = DataLoader(train_ds, batch_size=4,shuffle=True, num_workers=4)

    testloader = DataLoader(test_ds, batch_size=4,shuffle=True, num_workers=4)

    dataloaders = {"train": trainloader, "val": testloader}
    
    # dataset_pytorch can now be used in a data_loader
    """
    data_loader = torch.utils.data.DataLoader(dataset_pytorch, shuffle=True, batch_size=2)
    
    for batch in data_loader:
        data, label = batch
        print('[batch, channels, width, height]:', data.size())
        print('[batch]:', label)
        print('*' * 5)
    """

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            since_epoch = time.time()
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                inputs = Variable(inputs.type(Tensor))
                labels = Variable(labels.type(LongTensor))
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects / len(datasets[phase])
            
        time_elapsed_epoch = time.time() - since_epoch
        print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(phase, epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60))
                
        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 120)

if torch.cuda.device_count() > 1 and multiGPU:
    print("Using", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)

if use_gpu:
    model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
