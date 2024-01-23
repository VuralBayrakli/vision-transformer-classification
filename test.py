import sys
sys.path.append("/content/drive/MyDrive/vision_transformer_classification/vision_transformer_classification")
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

test_loader = DataLoader(test_dataset, batch_size=batch_size)

load_model = torch.load("/content/drive/MyDrive/vision_transformer_classification/vision_transformer_classification/model_vit.pth")

vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# Çıkış sınıflarının sayısını değiştir
vision_transformer.heads = nn.Linear(in_features=768, out_features=2, bias=True)

vision_transformer.load_state_dict(load_model)

accuracy = 0

# Sınıf sayısı
n_class = 2

# Her bir sınıf için doğru tahminleri ve toplam sayıları izleme
class_correct = np.zeros(n_class)
class_total = np.zeros(n_class)

# Modeli CPU'ya taşı
vision_transformer = vision_transformer
vision_transformer.eval()

# Modeli test et
for images, targets in test_loader:
    
    # Çıktıları al
    images = images
    target = targets
    outputs = vision_transformer(images)
    outputs = outputs
    
    # Olasılıklardan tahminleri al
    preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    preds = preds
    
    # Doğru tahminleri al
    correct_preds = (preds == targets).type(torch.FloatTensor)

    
    # Doğruluk hesapla ve biriktir
    accuracy += torch.mean(correct_preds).item() * 100
    
    # Her sınıf için test doğruluğunu hesapla
    for c in range(n_class):
        
        class_total[c] += (targets == c).sum()
        class_correct[c] += ((correct_preds) * (targets == c)).sum()

# Ortalama doğruluğu al
accuracy = accuracy / len(test_loader)

# Test kayıp istatistiklerini yazdır
print('accuracy: {:.6f}'.format(accuracy))