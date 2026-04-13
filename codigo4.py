import cv2
import numpy as np
import matplotlib.pyplot as plt

def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def psnr(img1, img2):
    erro = mse(img1, img2)
    if erro == 0:
        return 99
    return 10 * np.log10((255 ** 2) / erro)

f = cv2.imread("padrao_d.png", cv2.IMREAD_GRAYSCALE)
g_gray = cv2.imread("./resultados2/4partes.jpg", cv2.IMREAD_GRAYSCALE)

x, y = 0, 0
h, w = f.shape
g_crop = g_gray[y:y+h, x:x+w]

if g_crop.shape != f.shape:
    raise ValueError(f"Recorte inválido: {g_crop.shape} diferente de {f.shape}")

# ROI homogênea usada para estimar o ruído
#y1, y2 = 10, 120
#x1, x2 = 130, 250

#roi_f = f[y1:y2, x1:x2]
#roi_g = g_crop[y1:y2, x1:x2]

# Experimento 3: ruído aproximado na ROI
ruido_roi = g_crop.astype(np.int16) - f.astype(np.int16)

media = np.mean(ruido_roi)
desvio = np.std(ruido_roi)

print("Média do ruído na ROI:", media)
print("Desvio padrão do ruído na ROI:", desvio)

# Experimento 4: gerar ruído sintético com parâmetros da ROI
np.random.seed(42)
ruido_sintetico = np.random.normal(media, desvio, f.shape)

g_simulada = f.astype(np.float32) + ruido_sintetico
g_simulada = np.clip(g_simulada, 0, 255).astype(np.uint8)

# Comparação global
mse_global = mse(g_crop, g_simulada)
psnr_global = psnr(g_crop, g_simulada)

# Comparação na ROI
#roi_simulada = g_simulada[y1:y2, x1:x2]
#mse_roi = mse(roi_g, roi_simulada)
#psnr_roi = psnr(roi_g, roi_simulada)

print("\nComparação na imagem inteira:")
print("MSE global:", mse_global)
print("PSNR global:", psnr_global)

#print("\nComparação na ROI:")
#print("MSE ROI:", mse_roi)
#print("PSNR ROI:", psnr_roi)

# Visualização da ROI usada
#img_vis = cv2.cvtColor(g_crop, cv2.COLOR_GRAY2BGR)
#cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

#plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem ideal f(x,y)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(g_crop, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem real g(x,y)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(g_simulada, cmap="gray", vmin=0, vmax=255)
plt.title("Imagem simulada g_s(x,y)")
plt.axis("off")

#plt.subplot(2, 3, 4)
#plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
#plt.title("ROI usada na análise")
#plt.axis("off")

plt.subplot(2, 3, 4)
plt.hist(ruido_roi.ravel(), bins=50, color="gray", edgecolor="black")
plt.title("Histograma do ruído")
plt.xlabel("Valor do ruído")
plt.ylabel("Frequência")

plt.subplot(2, 3, 5)
diferenca = g_crop.astype(np.int16) - g_simulada.astype(np.int16)
plt.imshow(diferenca, cmap="gray")
plt.title("Diferença: g - g_s")
plt.axis("off")

plt.tight_layout()
plt.show()