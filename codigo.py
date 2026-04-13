import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread("padrao_d.png", cv2.IMREAD_GRAYSCALE)

g_gray = cv2.imread("./resultados2/4partes.jpg", cv2.IMREAD_GRAYSCALE)

x, y = 0, 0
h, w = f.shape

g_crop = g_gray[y:y+h, x:x+w]

if g_crop.shape != f.shape:
    raise ValueError(f"Recorte inválido: {g_crop.shape} diferente de {f.shape}")

d = g_crop.astype(np.int16) - f.astype(np.int16)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(f, cmap='gray', vmin=0, vmax=255)
plt.title("Imagem ideal f(x,y)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(g_crop, cmap='gray', vmin=0, vmax=255)
plt.title("Imagem real capturada g(x,y)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(d, cmap='seismic')
plt.title("Diferença d(x,y)=g-f")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()

#desenhar linha a linha
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(f, cmap="gray")
plt.axhline(64, color="red")
#plt.axvline(128, color="yellow")
plt.title("Imagem ideal")

plt.subplot(1,2,2)
plt.imshow(g_crop, cmap="gray")
plt.axhline(64, color="red")
#plt.axvline(128, color="yellow")
plt.title("Imagem real")

plt.show()

import numpy as np

# Escolha da região
linha_inicial = 140
coluna_inicial = 240
tamanho = 5

regiao_f = f[linha_inicial:linha_inicial+tamanho, coluna_inicial:coluna_inicial+tamanho]
regiao_g = g_crop[linha_inicial:linha_inicial+tamanho, coluna_inicial:coluna_inicial+tamanho]
regiao_dif = regiao_g.astype(np.int16) - regiao_f.astype(np.int16)

print("f(x,y):")
print(regiao_f)

print("\ng(x,y):")
print(regiao_g)

print("\nDiferença pixel a pixel g-f:")
print(regiao_dif)