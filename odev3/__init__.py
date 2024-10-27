import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Masaüstündeki kare ve daire resim dosyalarının yolunu ayarla
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

def edge_detection(image_path):
    # Görüntüyü gri tonlamaya çevirip Canny ile kenar tespiti yap
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Canny(image, 100, 200)

# Kare ve daire resimlerinin kenarlarını belirle
edges_square = edge_detection(os.path.join(desktop_path, "kare.png"))
edges_circle = edge_detection(os.path.join(desktop_path, "daire.png"))

def compute_gradients(image):
    # Sobel filtresi ile yatay ve dikey türevleri hesapla
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

# Kare ve daire resimleri için yatay ve dikey türevleri hesapla
grad_x_square, grad_y_square = compute_gradients(edges_square)
grad_x_circle, grad_y_circle = compute_gradients(edges_circle)

def compute_magnitude(grad_x, grad_y):
    # Pisagor teoremi ile büyüklük hesapla
    return np.sqrt(grad_x**2 + grad_y**2)

magnitude_square = compute_magnitude(grad_x_square, grad_y_square)
magnitude_circle = compute_magnitude(grad_x_circle, grad_y_circle)

plt.figure(figsize=(12, 8))

# Kare resim çıktıları
plt.subplot(2, 4, 1)
plt.imshow(edges_square, cmap='gray')
plt.title("Kare Kenarları")

plt.subplot(2, 4, 2)
plt.imshow(grad_x_square, cmap='gray')
plt.title("Kare Yatay Türev")

plt.subplot(2, 4, 3)
plt.imshow(grad_y_square, cmap='gray')
plt.title("Kare Dikey Türev")

plt.subplot(2, 4, 4)
plt.imshow(magnitude_square, cmap='gray')
plt.title("Kare Magnitute")

# Daire resim çıktıları
plt.subplot(2, 4, 5)
plt.imshow(edges_circle, cmap='gray')
plt.title("Daire Kenarları")

plt.subplot(2, 4, 6)
plt.imshow(grad_x_circle, cmap='gray')
plt.title("Daire Yatay Türev")

plt.subplot(2, 4, 7)
plt.imshow(grad_y_circle, cmap='gray')
plt.title("Daire Dikey Türev")

plt.subplot(2, 4, 8)
plt.imshow(magnitude_circle, cmap='gray')
plt.title("Daire Magnitute")

plt.tight_layout()
plt.show()
