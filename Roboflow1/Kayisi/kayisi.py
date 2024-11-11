import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv5 modelini başlat
model = YOLO('yolov5n.pt')  # YOLOv5 Nano modeli

# Roboflow API anahtarı ile projeyi indir
rf = Roboflow(api_key="5etQ4r2F3oiRc247YEuK")  # Roboflow API anahtarınızı buraya ekleyin
project = rf.workspace().project("Odev1")  # Çalışma alanı ve proje adını güncelleyin
version = project.version(1)  # İndirmek istediğiniz proje versiyonunu belirtin
dataset = version.download("yolov5")  # YOLOv5 formatında veri setini indirin

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
# Modeli eğit
if __name__ == "__main__":
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)  # Eğitim verilerinin konumunu ve eğitim sayısını ayarlayın
