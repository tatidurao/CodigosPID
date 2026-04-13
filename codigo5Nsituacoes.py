import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# FUNÇÃO PARA ANALISAR BORDA COM ESF E LSF NORMALIZADAS
# =========================================================
def analisar_borda_normalizada(
    caminho_imagem,
    titulo_imagem,
    modo="horizontal",   # "horizontal" ou "vertical"
    linha_y=50,          # usado no perfil horizontal
    coluna_x=128,        # usado no perfil vertical
    x_inicio=0,
    x_fim=None,
    y_inicio=0,
    y_fim=None
):
    """
    Analisa uma borda em uma imagem em escala de cinza.

    modo = "horizontal" -> extrai uma linha da imagem
    modo = "vertical"   -> extrai uma coluna da imagem

    Retorna ESF, LSF e medidas da transição.
    """

    # -----------------------------
    # CARREGAR IMAGEM
    # -----------------------------
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")

    h, w = img.shape

    if x_fim is None:
        x_fim = w
    if y_fim is None:
        y_fim = h

    # -----------------------------
    # EXTRAÇÃO DO PERFIL
    # -----------------------------
    if modo == "horizontal":
        perfil = img[linha_y, x_inicio:x_fim].astype(np.float32)
        eixo = np.arange(len(perfil))

    elif modo == "vertical":
        perfil = img[y_inicio:y_fim, coluna_x].astype(np.float32)
        eixo = np.arange(len(perfil))

    else:
        raise ValueError("O modo deve ser 'horizontal' ou 'vertical'.")

    # -----------------------------
    # NORMALIZAÇÃO DA ESF
    # -----------------------------
    imin = np.min(perfil)
    imax = np.max(perfil)

    if imax == imin:
        raise ValueError("O perfil possui intensidade constante; não é possível normalizar.")

    esf_norm = (perfil - imin) / (imax - imin)

    # -----------------------------
    # NÍVEIS DE 10% E 90%
    # -----------------------------
    idx_10 = np.argmin(np.abs(esf_norm - 0.10))
    idx_90 = np.argmin(np.abs(esf_norm - 0.90))
    Ax = abs(idx_90 - idx_10)

    print("\n" + "="*65)
    print(f"ANÁLISE: {titulo_imagem}")
    print("="*65)
    print(f"Intensidade mínima: {imin}")
    print(f"Intensidade máxima: {imax}")
    print(f"X10% = {idx_10}")
    print(f"X90% = {idx_90}")
    print(f"Ax = |X90 - X10| = {Ax}")

    # -----------------------------
    # LSF = derivada da ESF
    # -----------------------------
    lsf = np.gradient(esf_norm)

    # Normalizar LSF
    max_lsf = np.max(np.abs(lsf))
    if max_lsf != 0:
        lsf_norm = lsf / max_lsf
    else:
        lsf_norm = lsf

    # -----------------------------
    # MOSTRAR IMAGEM COM REGIÃO ANALISADA
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(f"{titulo_imagem} - Região analisada")

    if modo == "horizontal":
        plt.axhline(linha_y, color="red", label=f"y = {linha_y}")
        plt.axvline(x_inicio, color="yellow", linestyle="--", label="início/fim do recorte")
        plt.axvline(x_fim - 1, color="yellow", linestyle="--")
    else:
        plt.axvline(coluna_x, color="red", label=f"x = {coluna_x}")
        plt.axhline(y_inicio, color="yellow", linestyle="--", label="início/fim do recorte")
        plt.axhline(y_fim - 1, color="yellow", linestyle="--")

    plt.legend()
    plt.axis("on")
    plt.show()

    # -----------------------------
    # GRÁFICO ESF NORMALIZADA
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(eixo, esf_norm, label="ESF normalizada")
    plt.axhline(0.10, linestyle="dashed", label="10%")
    plt.axhline(0.90, linestyle="dashdot", label="90%")
    plt.axvline(idx_10, linestyle="dotted", label=f"X10 = {idx_10}")
    plt.axvline(idx_90, linestyle="--", label=f"X90 = {idx_90}")
    plt.title(f"ESF normalizada - {titulo_imagem}")
    plt.xlabel("Posição no perfil")
    plt.ylabel("Intensidade normalizada")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # GRÁFICO LSF NORMALIZADA
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(eixo, lsf_norm, label="LSF normalizada")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title(f"LSF normalizada - {titulo_imagem}")
    plt.xlabel("Posição no perfil")
    plt.ylabel("Amplitude normalizada")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "perfil_original": perfil,
        "esf_norm": esf_norm,
        "lsf_norm": lsf_norm,
        "imin": imin,
        "imax": imax,
        "x10": idx_10,
        "x90": idx_90,
        "Ax": Ax
    }


# =========================================================
# SITUAÇÃO 1 - IMAGEM DE 2 PARTES
# Perfil vertical
# =========================================================
resultado_2_partes = analisar_borda_normalizada(
    caminho_imagem="./resultados2/2partes.jpg",
    titulo_imagem="Situação 1 - Imagem de 2 partes (perfil vertical)",
    modo="vertical",
    coluna_x=128,   # ajuste se necessário
    y_inicio=0,
    y_fim=256
)

# =========================================================
# SITUAÇÃO 2 - IMAGEM DE 4 PARTES
# Perfil horizontal
# =========================================================
resultado_4_partes = analisar_borda_normalizada(
    caminho_imagem="./resultados2/4partes.jpg",
    titulo_imagem="Situação 2 - Imagem de 4 partes (perfil horizontal)",
    modo="horizontal",
    linha_y=50,
    x_inicio=10,
    x_fim=250
)

# =========================================================
# SITUAÇÃO 3 - RECORTE LOCAL DA BORDA
# Linha horizontal y=50 e recorte entre x=100 e x=150
# =========================================================
resultado_recorte_local = analisar_borda_normalizada(
    caminho_imagem="./resultados2/4partes.jpg",
    titulo_imagem="Situação 3 - Recorte local da borda (x=100 até x=150)",
    modo="horizontal",
    linha_y=50,
    x_inicio=100,
    x_fim=150
)