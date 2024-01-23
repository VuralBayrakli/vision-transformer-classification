# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:26:44 2024

@author: VuralBayraklii
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, key_dim=64):
        super(SelfAttention, self).__init__()
        
        self.embedding_dim = embedding_dim   # D -> gömme boyutu
        self.key_dim = key_dim               # D_h -> anahtar, sorgu, değer boyutu
        
        # U_kqv ağırlık matrisi
        self.W = nn.Parameter(torch.randn(embedding_dim, 3*key_dim))
    
    def forward(self, x):
        key_dim = self.key_dim

        # sorgu, anahtar ve değer projeksiyonlarını elde et
        qkv = torch.matmul(x, self.W)
        # x_p(1,65,768) W(768,3*64) = (1,65,3*64)
        # sorgu, anahtar, değeri elde et
        q = qkv[:, :, :key_dim] # (1,65,64)
        k = qkv[:, :, key_dim:key_dim*2 ] #(1,65,64) --- (1,65,64)*(1,64,65) = (1,65,65)*(1,65,64) = (1,65,64)
        v = qkv[:, :, key_dim*2:]
        
        # tüm sorguların tüm anahtarlarla dot ürününü hesapla  
        k_T = torch.transpose(k, -2, -1)   # anahtarın transpozunu al
        dot_products = torch.matmul(q, k_T) 
        # 12 tane (1,65,64) --- (1,65,768)
        # her birini √Dh ile böl
        scaled_dot_products = dot_products / np.sqrt(key_dim)
        
        # dikkat ağırlıklarını elde etmek için bir softmax fonksiyonu uygula -> A
        attention_weights = F.softmax(scaled_dot_products, dim=1)
        # self.attention_weights = [w.detach().numpy() for w in attention_weights]

        # ağırlıklı değerleri al
        weighted_values = torch.matmul(attention_weights, v)
        
        # ağırlıklı değerleri döndür
        return weighted_values
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_heads = num_heads            # başlık sayısını (k) ayarla
        self.embedding_dim = embedding_dim    # boyutu ayarla
        
        assert embedding_dim % num_heads == 0   # boyut, başlık sayısına bölünebilmelidir
        self.key_dim = embedding_dim // num_heads   # anahtar, sorgu ve değer boyutunu ayarla
        
        # self-attention'ları başlat
        self.attention_list = [SelfAttention(embedding_dim, self.key_dim) for _ in range(num_heads)]
        self.multi_head_attention = nn.ModuleList(self.attention_list) 
        
        # U_msa ağırlık matrisini başlat
        self.W = nn.Parameter(torch.randn(num_heads * self.key_dim, embedding_dim))
        
    def forward(self, x):
        # her bir başlığın kendine dikkat skorlarını hesapla 
        attention_scores = [attention(x) for attention in self.multi_head_attention]

        # dikkatleri birleştir(son boyut)
        Z = torch.cat(attention_scores, -1)
        
        # çoklu başlı dikkat skorunu hesapla
        attention_score = torch.matmul(Z, self.W)
        # Z (1,65,768) W(768,768) --(1,65,768)
        
        return attention_score

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=3072):
        super(MultiLayerPerceptron, self).__init__()
        
        # MLP katmanlarını tanımla
        self.mlp = nn.Sequential(
                            nn.Linear(embedding_dim, hidden_dim),  # Giriş katmanı
                            nn.GELU(),  # GELU aktivasyonu
                            nn.Linear(hidden_dim, embedding_dim)  # Çıkış katmanı
                   )
        
    def forward(self, x):
        # Çok katmanlı perceptron'dan geçir
        x = self.mlp(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, hidden_dim=3072, dropout_prob=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Çoklu başlı kendine dikkat ve MLP katmanlarını başlat
        self.MSA = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.MLP = MultiLayerPerceptron(embedding_dim, hidden_dim)
        
        # Layer normalization katmanlarını başlat
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout katmanlarını başlat
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # Dropout uygula
        out_1 = self.dropout1(x)
        # Layer normalization uygula
        out_2 = self.layer_norm1(out_1)
        # Çoklu başlı kendine dikkat uygula
        msa_out = self.MSA(out_2)
        # Dropout uygula
        out_3 = self.dropout2(msa_out)
        # Rezidüel bağlantı uygula
        res_out = x + out_3
        # Layer normalization uygula
        out_4 = self.layer_norm2(res_out)
        # MLP çıkışını hesapla
        mlp_out = self.MLP(out_4)
        # Dropout uygula
        out_5 = self.dropout3(mlp_out)
        # Rezidüel bağlantı uygula
        output = res_out + out_5
        
        return output


class MLPHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=10, fine_tune=False):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        
        if not fine_tune:
            # Tanh aktivasyon fonksiyonlu gizli katman
            self.mlp_head = nn.Sequential(
                                    nn.Linear(embedding_dim, 3072),  # Gizli katman
                                    nn.Tanh(),
                                    nn.Linear(3072, num_classes)    # Çıkış katmanı
                            )
        else:
            # Tek lineer katman
            self.mlp_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.mlp_head(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, image_size=224, channel_size=3, 
                     num_layers=12, embedding_dim=768, num_heads=12, hidden_dim=3072, 
                            dropout_prob=0.1, num_classes=10, pretrain=True):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size 
        self.channel_size = channel_size 
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        
        # Görüntünün kaç adet yama içerdiğini al
        self.num_patches = int(image_size ** 2 / patch_size ** 2)   # height * width / patch size ^ 2   
        
        # Boyutlarını düzenleme için eğitilebilir lineer projeksiyon (ağırlık matrisi E)
        self.W = nn.Parameter(
                    torch.randn( patch_size * patch_size * channel_size, embedding_dim))
        
        # Pozisyon gömme (E_pos)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embedding_dim))
        
        # Öğrenilebilir sınıf simgesi gömme (x_class)
        self.class_token = nn.Parameter(torch.rand(1, embedding_dim))
        
        # Transformer önceleyici katmanlarını yığınla (stack)
        transformer_encoder_list = [
            TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob) 
                    for _ in range(num_layers)] 
        self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)
        
        # MLP başı
        self.mlp_head = MLPHead(embedding_dim, num_classes)
        
    def forward(self, x):
        # Yama boyutu ve kanal sayısını al
        P, C = self.patch_size, self.channel_size
        
        # Görüntüyü yamalara böl
        patches = x.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(patches.size(0), -1, C * P * P).float()
        
        # Yama gömme işlemi
        patch_embeddings = torch.matmul(patches , self.W)
        
        # Sınıf simgesini ekleyin
        batch_size = patch_embeddings.shape[0]
        patch_embeddings = torch.cat((self.class_token.repeat(batch_size, 1, 1), patch_embeddings), 1)
        
        # Pozisyon gömme ekleyin
        patch_embeddings = patch_embeddings + self.pos_embedding
        
        # Yama gömülerini bir yığın Transformer önceleyiciye besleyin
        transformer_encoder_output = self.transformer_encoder_layers(patch_embeddings)
        
        # Encoder çıkışından [class] simgesini çıkar
        output_class_token = transformer_encoder_output[:, 0]
        
        # Sınıflandırma için mlp başından geçirin
        y = self.mlp_head(output_class_token)
        
        return y
