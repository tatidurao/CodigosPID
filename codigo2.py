import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread("padrao_e.png", cv2.IMREAD_GRAYSCALE)

g_gray = cv2.imread("./resultados2/16partes2.jpg", cv2.IMREAD_GRAYSCALE)

x, y = 0, 0
h, w = f.shape

g_crop = g_gray[y:y+h, x:x+w]


# EXPERIMENTO 2
# PERFIL DE BORDA
linha = 64  # linha que atravessa a borda vertical

perfil_f = f[linha, :]
perfil_g = g_crop[linha, :]   # usar a imagem capturada original, sem normalizar

print("Valores da linha na imagem ideal f:")
print(perfil_f)

print("\nValores da linha na imagem capturada g:")
print(perfil_g)

plt.figure(figsize=(10,4))
plt.plot(perfil_f, label="Ideal")
plt.plot(perfil_g, label="Capturada")
plt.title("Perfil de intensidade na borda")
plt.xlabel("Posição x")
plt.ylabel("Intensidade")
plt.legend()
plt.grid()
plt.show()