import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread("padrao_b.png", cv2.IMREAD_GRAYSCALE)

g_gray = cv2.imread("./resultados2/16partes2.jpg", cv2.IMREAD_GRAYSCALE)

x, y = 0, 0
h, w = f.shape

g_crop = g_gray[y:y+h, x:x+w]

#y1, y2 = 130, 250
#x1, x2 = 10, 130

#roi_f = f[y1:y2, x1:x2]
#roi_g = g_crop[y1:y2, x1:x2]

ruido_roi = g_crop.astype(np.int16) - f.astype(np.int16)


plt.figure(figsize=(7,4))
plt.hist(ruido_roi.ravel(), bins=50, color='gray', edgecolor='black')
plt.title("Histograma do ruído aproximado imagem completa")
plt.xlabel("Valor de n(x,y)")
plt.ylabel("Frequência")
plt.grid()
plt.show()