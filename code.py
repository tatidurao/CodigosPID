import cv2
import numpy as np

tamanho = 256


# imagem começa com 200 - cinza claro
a = np.full((tamanho, tamanho), 200, dtype=np.uint8)

# metade de cima clara, metade de baixo mais escuro
b = np.full((tamanho, tamanho), 200, dtype=np.uint8)
b[128:256, :] = 160

# 4 retângulos internos
c = np.full((tamanho, tamanho), 210, dtype=np.uint8)
c[128:256, :] = 180
c[40:70, 40:80] = 180
c[40:70, 120:160] = 180
c[160:190, 40:80] = 230
c[160:190, 120:160] = 230

# dividido em 4 partes 2x2
d = np.zeros((tamanho, tamanho), dtype=np.uint8)

# Quadrante superior esquerdo
d[0:128, 0:128] = 140
# Quadrante superior direito
d[0:128, 128:256] = 170
# Quadrante inferior esquerdo
d[128:256, 0:128] = 210
# Quadrante inferior direito
d[128:256, 128:256] = 235

# grade 4x4
e = np.zeros((tamanho, tamanho), dtype=np.uint8)

#  64x64 pixels porque 256 / 4 = 64
bloco = 64

# Valores de cinza da grade
tons = [
    [110, 130, 150, 170],
    [140, 160, 180, 200],
    [170, 190, 210, 220],
    [200, 215, 230, 240]
]

for linha in range(4):
    for coluna in range(4):
        inicio_linha = linha * bloco
        fim_linha = (linha + 1) * bloco
        inicio_coluna = coluna * bloco
        fim_coluna = (coluna + 1) * bloco
        
        e[inicio_linha:fim_linha, inicio_coluna:fim_coluna] = tons[linha][coluna]


cv2.imwrite("a.png", a)
cv2.imwrite("b.png", b)
cv2.imwrite("c.png", c)
cv2.imwrite("d.png", d)
cv2.imwrite("e.png", e)


cv2.imshow("A", a)
cv2.imshow("B", b)
cv2.imshow("C", c)
cv2.imshow("D", d)
cv2.imshow("E", e)

cv2.waitKey(0)
cv2.destroyAllWindows()

