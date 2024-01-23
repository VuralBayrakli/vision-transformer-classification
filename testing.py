import argparse
from PIL import Image
import torch
from torchvision import transforms
from model import VisionTransformer
import torchvision.models as models
import torch.nn as nn

def load_model(model_path):
    # Eğitilmiş modelinizi yükleyin
    load_model = torch.load(model_path, map_location=torch.device('cpu'))

    vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Çıkış sınıflarının sayısını değiştir
    vision_transformer.heads = nn.Linear(in_features=768, out_features=2, bias=True)

    vision_transformer.load_state_dict(load_model)

    vision_transformer.eval()

    return vision_transformer

def preprocess_image(image_path):
    # Giriş görüntüsünü ön işleme adımlarını uygulayın
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Batch boyutunu ekleyin
    return image

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Inference')
    parser.add_argument('--model', type=str, required=True, help='Eğitilmiş modelin yolu')
    parser.add_argument('--image', type=str, required=True, help='Çalıştırılacak görüntünün yolu')
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path)
    model = model.to(device)

    input_image = preprocess_image(image_path)
    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)

    # Sınıflar üzerinde softmax uygulayarak tahminleri alın
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # En yüksek olasılığa sahip sınıfı belirleyin
    predicted_class = torch.argmax(probabilities).item()
    if predicted_class == 0:
        print("Görüntü sınıfı: Ateş Var, Olasılık: {:.4f}".format(probabilities[0]))

    else:
        print("Görüntü sınıfı: Ateş Yok, Olasılık: {:.4f}".format(probabilities[1]))

if __name__ == "__main__":
    main()
