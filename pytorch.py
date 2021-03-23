# https://www.kaggle.com/puneet6060/intel-image-classification
from __future__ import print_function, division
import os
import torchvision.models as models
import torch
from torch.nn import CrossEntropyLoss, Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, Softmax, Module
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.utils.data as data
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() \
      else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list,tuple)) \
      else data.to(device, non_blocking=True)

def batches_to_device(data_loader, device):
    for batch in data_loader:
        yield to_device(batch, device)

def save_checkpoints(epoch: int, model: Module, optimizer: SGD, loss: _Loss, path: str):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, path)

def save_model(model: Module, model_name: str, dir: str):
  torch.save(model, f'{dir}{model_name}.pt')

TRAIN_FOLDER_PATH = "E:/Image Datasets/fruits-360/Training"
TEST_FOLDER_PATH = "E:/Image Datasets/fruits-360/Test"

def load_fruits_360_data(batch=30, root_path="E:/Image Datasets/Intel Scenes/"):
    transform = transforms.Compose([transforms.Resize((150, 150)),
                                    transforms.ToTensor()])

    dataset = ImageFolder(root=TRAIN_FOLDER_PATH, transform=transform)
    
    torch.manual_seed(55)
    train_dataset, val_dataset = random_split(dataset, [11928, len(dataset) - 11928])

    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,  num_workers=4)

    test_data = ImageFolder(root=TEST_FOLDER_PATH, transform=transform)
    test_data_loader  = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader




# print(resformer(data))

"""
    T R A I N N I N G
"""
train_loader, val_loader, test_loader = load_fruits_360_data()

loss = CrossEntropyLoss()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch 
    images, labels = images.cuda(), labels.cuda()
    out = model.forward(images)
    cross_entropy = CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: Module, val_set: DataLoader):
    outputs = [validation_step(model, batch) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def train(epochs_no: int, model: Module, train_set: DataLoader, val_set: DataLoader):
    history = []
    
    # TODO Read about optimizer optimizer = opt_func(model.parameters(), lr)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs_no):
        """  Training Phase """ 
        for batch in train_set:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            out = model.forward(inputs)
            curr_loss = loss(out, labels)
            curr_loss.backward()
            optimizer.step()
            


        """ Validation phase """
        result = evaluate(model, val_set)
        print(result)
        history.append(result)
        if epoch % 10 == 0 :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': curr_loss,
                }, 'E:/dxlat/Training stats/model.pt')
    return history

def train_model(epochs_no, model_to_train, name: str):
    device = get_default_device()

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(model_to_train, device)

    train(epochs_no, model, train_loader, val_loader)

    torch.save(model, f'E:/dxlat/Training stats/{name}.pt')

if __name__ == "__main__":
    device = get_default_device()
    
    res50 = models.resnet50(pretrained=True)
    res50.train()
    # train_model(100, res50, 'modelResFruits')

    print("resnet 50")
    checkpoint = torch.load('E:/dxlat/Training stats/model.pt')
    import copy
    res50 = copy.deepcopy(res50)
    res50.load_state_dict(checkpoint['model_state_dict'])
    res50 = to_device(res50, device)
    total_correct = 0
    total_data = 22688
    with torch.no_grad():
        for batch in test_loader:
            i, l = batch
            i, l = i.cuda(), l.cuda()
            out = res50(i)
            _, predicted = torch.max(out, 1)
            c = (predicted == l).squeeze()
            for v in c:
                if v :
                    total_correct += 1
           
    
    print(f'Accuracy of {100 * total_correct/total_data}')

