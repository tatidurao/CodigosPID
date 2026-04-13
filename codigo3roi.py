import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread("padrao_b.png", cv2.IMREAD_GRAYSCALE)

g_gray = cv2.imread("./resultados2/2partes.jpg", cv2.IMREAD_GRAYSCALE)

x, y = 0, 0
h, w = f.shape
g_crop = g_gray[y:y+h, x:x+w]

#y1, y2 = 50, 120
#x1, x2 = 10, 250

y1, y2 = 130, 250
x1, x2 = 10, 130

roi_f = f[y1:y2, x1:x2]
roi_g = g_crop[y1:y2, x1:x2]

ruido_roi = roi_g.astype(np.int16) - roi_f.astype(np.int16)

img_vis = cv2.cvtColor(g_crop, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title("Região usada para análise do ruído")
plt.axis("off")
plt.show()

print("Amostra da ROI da imagem ideal f:")
print(roi_f[:5, :5])

print("\nAmostra da ROI da imagem real g:")
print(roi_g[:5, :5])

print("\nAmostra do ruído aproximado n(x,y):")
print(ruido_roi[:5, :5])

for i in range(5):
    for j in range(5):
        print(f"Pixel ({i},{j}) -> f={roi_f[i,j]}, g={roi_g[i,j]}, n={ruido_roi[i,j]}")