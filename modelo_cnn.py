# modelo_cnn.py

import tensorflow as tf
import numpy as np
import cv2

# O tamanho que você usou para treinar sua CNN
TAMANHO_IMAGEM_CNN = (128, 128) 

def criar_e_carregar_modelo(caminho_modelo='modelo_inspecao_visual.h5'):
    """
    Cria uma CNN simples e tenta carregar os pesos pré-treinados.
    Se o modelo existir, ele será carregado.
    Você deve treinar e salvar seu modelo com este nome primeiro!
    """
    try:
        model = tf.keras.models.load_model(caminho_modelo)
        print("Modelo CNN carregado com sucesso.")
    except:
        print("Aviso: Modelo CNN não encontrado. Criando modelo simples (NÃO TREINADO).")
        # Criação de um modelo placeholder simples
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANHO_IMAGEM_CNN[0], TAMANHO_IMAGEM_CNN[1], 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid') # 1 para classificação binária (Defeito/Sem Defeito)
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    return model

# Carrega o modelo uma vez para não recarregar a cada inspeção
CNN_MODEL = criar_e_carregar_modelo()


def inspecao_visual_cnn(image_path, model=CNN_MODEL):
    """
    Usa o modelo CNN para classificar a imagem como defeituosa ou não.

    Args:
        image_path (str): Caminho para a imagem do componente.
        model (tf.keras.Model): O modelo CNN carregado.

    Returns:
        tuple: (bool: True se APROVADO (sem defeito), False se REPROVADO, str: Mensagem de status)
    """
    image = cv2.imread(image_path)
    if image is None:
        return False, "Erro ao carregar imagem para CNN."

    # Pré-processamento para a CNN
    img_resized = cv2.resize(image, TAMANHO_IMAGEM_CNN)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0 # Normalização
    
    # Previsão
    prediction = model.predict(img_array)[0]

    # Interpretação do resultado (Ajuste o limiar 0.5 conforme seu treino)
    if prediction < 0.5:
        # Assumindo que < 0.5 significa "Sem Defeito"
        return True, f"Visual OK (Score: {prediction[0]:.2f})"
    else:
        # Assumindo que >= 0.5 significa "DEFEITO"
        return False, f"REPROVADO: Defeito Visual Detectado (Score: {prediction[0]:.2f})"