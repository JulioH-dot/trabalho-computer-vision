# inspecao_dimensional.py

import cv2
import numpy as np

# --- Parâmetros de Calibração e Padrão ---
LIMIAR_SEGMENTACAO = 150
LARGURA_MIN_PADRAO_PIXEL = 1000
LARGURA_MAX_PADRAO_PIXEL = 4500
ALTURA_MIN_PADRAO_PIXEL = 500
ALTURA_MAX_PADRAO_PIXEL = 1200
#

def inspecao_dimensional(image_path):
    """
    Realiza a segmentação e a medição dimensional de um componente.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False, "Erro ao carregar imagem."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except:
        # Se Otsu falhar (imagem com pouca variação, etc.), voltamos para o limiar fixo.
        _, thresh = cv2.threshold(gray, LIMIAR_SEGMENTACAO, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maior_area = 0
    contorno_principal = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filtro de área mínima: 5000 para ignorar ruído.
        if area > 5000: 
            if area > maior_area:
                maior_area = area
                contorno_principal = cnt

    dimensao_aprovada = False
    status_msg = "Nenhum componente principal detectado (Área < 5000)."

    if contorno_principal is not None:
        x, y, w, h = cv2.boundingRect(contorno_principal)


        is_largura_ok = LARGURA_MIN_PADRAO_PIXEL <= w <= LARGURA_MAX_PADRAO_PIXEL
        is_altura_ok = ALTURA_MIN_PADRAO_PIXEL <= h <= ALTURA_MAX_PADRAO_PIXEL

        if is_largura_ok and is_altura_ok:
            dimensao_aprovada = True
            status_msg = f"APROVADO: Largura ({w}px) e Altura ({h}px) dentro do padrão."
        else:
            dimensao_aprovada = False
            status_msg = f"REPROVADO: Largura ({w}px) ou Altura ({h}px) fora do padrão."

    return dimensao_aprovada, status_msg