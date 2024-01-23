
# Vision Transformer ile Ateş Sınıflandırma

Bu projede Vision Transformer mimarisi kullanılmıştır. Vision Transformer, büyük boyutlu görüntüleri daha etkili bir şekilde temsil etmek için dikkat mekanizmalarını içerir.

Proje, ateş sınıflandırması üzerine olarak eğitilmiş bir Vision Transformer modelini içerir. Bu model, ateş veri kümesi üzerinde eğitilmiş ve ateşle ilişkilendirilmiş farklı durumları başarıyla tanıma yeteneğine sahiptir.


## Projeyi yükleyin

```bash
    git clone https://github.com/VuralBayrakli/vision-transformer-classification.git
```

## Modeli indirin
Eğitilmiş modeli [buradan](https://drive.google.com/file/d/1c4wG1VGJSpiwkxuRnGVt2keGUb59T1oT) indirin ve projenin kök dizinine yerleştirin.


## Gerekli kütüphaneleri yükleyin
```bash
    pip install -r requirements.txt
```

## Modeli kullanın
```bash
    python testing.py --model <model_path> --image <image_path>
```

## Örnekler

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-classification/blob/master/ss/ss1.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-classification/blob/master/ss/ss2.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-classification/blob/master/ss/ss3.png)
