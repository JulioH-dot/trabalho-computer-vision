# treinar_cnn.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modelo_cnn import criar_e_carregar_modelo, TAMANHO_IMAGEM_CNN 
import os

# --- Configurações de Treinamento ---
DATASET_DIR = 'dataset'
BATCH_SIZE = 100  # Número de imagens processadas por vez
NUM_EPOCHS = 100  # Número de vezes que o modelo verá todo o dataset
MODEL_OUTPUT_NAME = 'modelo_inspecao_visual.h5'


def treinar_modelo():
    # 1. Pré-processamento e Aumento de Dados
    # Normaliza as imagens e define técnicas de aumento de dados
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalização obrigatória
        rotation_range=20, # Rotação leve para aumentar a diversidade
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # O conjunto de validação só precisa de normalização
    val_datagen = ImageDataGenerator(rescale=1./255)

    # 2. Carregamento dos Dados
    print("Carregando dados de TREINAMENTO...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=TAMANHO_IMAGEM_CNN, # Ex: (128, 128)
        batch_size=BATCH_SIZE,
        class_mode='binary' # Classificação APROVADO/REPROVADO (2 classes)
    )

    print("Carregando dados de VALIDAÇÃO...")
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'validation'),
        target_size=TAMANHO_IMAGEM_CNN,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 3. Criação/Carregamento do Modelo
    model = criar_e_carregar_modelo(caminho_modelo='placeholder.h5') # Carrega ou cria a estrutura

    # 4. Treinamento
    print("\n--- INICIANDO TREINAMENTO ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # 5. Salvar o Modelo Treinado
    model.save(MODEL_OUTPUT_NAME)
    print(f"\nModelo treinado salvo como: {MODEL_OUTPUT_NAME}")

if __name__ == '__main__':
    # Verifique se o diretório do dataset existe
    if not os.path.isdir(DATASET_DIR):
        print(f"Erro: O diretório do dataset '{DATASET_DIR}' não foi encontrado.")
        print("Crie a estrutura de pastas conforme o item 2 (Estrutura do Dataset).")
    else:
        treinar_modelo()