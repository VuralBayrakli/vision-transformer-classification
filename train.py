# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:30:44 2024

@author: VuralBayraklii
"""


from model import VisionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import os
import numpy as np

class Dataset:
    def __init__(self, path):
        self.path = path
        self.transform = transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def create_dataset(self):
        train_dataset = datasets.ImageFolder(root=os.path.join(self.path, "train"), transform=self.transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(self.path, "test"), transform=self.transform)
        
        return train_dataset, test_dataset
        
path_content = "/content/drive/MyDrive/vision_transformer_classification/vision_transformer_classification"
train_dataset, test_dataset = Dataset(os.path.join(path_content, "dataset")).create_dataset()

# Mini-batch boyutu
batch_size = 16

# Doğrulama seti olarak kullanılacak eğitim setinin yüzdesi
valid_size = 0.2

# Doğrulama için kullanılacak eğitim seti indislerini al
train_size = len(train_dataset)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor(valid_size * train_size))
train_idx, valid_idx = indices[split:], indices[:split]

# Eğitim ve doğrulama toplu alımları almak için örnekleyicileri tanımla
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Veri yükleyicilerini hazırla
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Eğitim setindeki tüm örnek etiketlerini al
train_labels = [labels for i, (images, labels) in enumerate(train_loader)]
train_labels = torch.cat((train_labels), 0)
train_labels_count = train_labels.unique(return_counts=True)

# Eğitim setindeki her sınıfın örnek sayısını yazdır
print('Eğitim veri setindeki sınıf başına örnek sayısı:\n')
for label, count in zip(train_labels_count[0], train_labels_count[1]):
    print('\t {}: {}'.format(label, count))

# Test setindeki tüm örnek etiketlerini al
test_labels = [labels for i, (images, labels) in enumerate(test_loader)]
test_labels = torch.cat((test_labels), 0)
test_labels_count = test_labels.unique(return_counts=True)

print()
print('Test veri setindeki sınıf başına örnek sayısı:\n')
for label, count in zip(test_labels_count[0], test_labels_count[1]):
    print('\t {}: {}'.format(label, count))
    
    
"""vision_transformer = VisionTransformer(patch_size, image_size, channel_size, 
                        n_layer, embedding_dim, n_head, hidden_dim, dropout_prob, n_class)"""
vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# Çıkış sınıflarının sayısını değiştir
vision_transformer.heads = nn.Linear(in_features=768, out_features=2, bias=True)

# Modelin parametrelerini dondurma
# Tüm parametreleri dondur
for p in vision_transformer.parameters():
    p.requires_grad = False

# Sınıflandırma başlığının ağırlıklarını eğitmek üzere parametrelerin dondurulmasını kaldır
for p in vision_transformer.heads.parameters():
    p.requires_grad = True


# Kayıp fonksiyonunu belirt
criterion = nn.CrossEntropyLoss()

# Optimizasyon algoritmasını tanımla
# Sadece requires_grad=True olan parametreleri eğit
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vision_transformer.parameters()), lr=0.0001)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()


def train():
    # Train model 

    # number of epochs
    n_epoch = 10
    
    # number of iterations to save model
    n_step=100
    
    train_loss_list, valid_loss_list = [], []
    
    # move model to GPU
    if train_on_gpu:
        vision_transformer.to('cuda')
    
    # prepare model for training
    vision_transformer.train()
    
    for e in range(n_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        
        # get batch data
        for i, (images, targets) in enumerate(train_loader):
            
            # move to gpu if available
            if train_on_gpu:
                images, targets = images.to('cuda'), targets.to('cuda')
            
            # clear grad
            optimizer.zero_grad()
            
            # feedforward data
            outputs = vision_transformer(images)
            
            # calculate loss
            loss = criterion(outputs, targets)
            
            # backward pass, calculate gradients
            loss.backward()
            
            # update weights
            optimizer.step()
            
            # track loss
            train_loss += loss.item()
            model_path = "/content/drive/MyDrive/vision_transformer_classification/vision_transformer_classification/model_vit.pth"
            # save the model parameters
            if i % n_step == 0:
              torch.save(vision_transformer.state_dict(), model_path)

        torch.save(vision_transformer.state_dict(), model_path)
        # set model to evaluation mode
        vision_transformer.eval()
        
        # validate model
        for images, targets in valid_loader:
            
            # move to gpu if available
            if train_on_gpu:
                images = images.to('cuda')
                targets = targets.to('cuda')
            
            # turn off gradients
            with torch.no_grad():
                
                outputs = vision_transformer(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                
        # set model back to trianing mode
        vision_transformer.train()
        
        # get average loss values
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        # output training statistics for epoch
        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'
                      .format( (e+1), train_loss, valid_loss))
        


if __name__ == '__main__':
    train()