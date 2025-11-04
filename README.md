# 🤖 Visão Computacional para Inspeção de Qualidade em Linha de Produção

## Projeto de Implementação de Sistema Híbrido de Inspeção de Componentes Eletrônicos

Este repositório contém o código-fonte de um protótipo de sistema de inspeção automatizada, desenvolvido para um desafio de controle de qualidade em uma linha de produção de componentes eletrônicos.

A solução implementada adota uma **arquitetura híbrida**, combinando a precisão do **OpenCV** (Visão Computacional Clássica) para verificações dimensionais e a capacidade de generalização do **Deep Learning** (TensorFlow/CNN) para detecção de defeitos visuais complexos.

## 🎯 Arquitetura e Objetivos

O sistema é orquestrado pelo `main.py` e executa duas etapas sequenciais:

1.  **Inspeção Dimensional:** Avalia se o componente está dentro das tolerâncias de largura e altura, usando detecção de contornos e Bounding Box.
2.  **Inspeção Visual:** Classifica a integridade estética do componente (Com Defeito / Sem Defeito) usando um modelo de Rede Neural Convolucional (CNN) treinado.

O veredito final é **APROVADO** somente se ambas as inspeções forem concluídas com sucesso.

## ⚙️ Tecnologias Principais

| Tecnologia | Finalidade no Projeto |
| :--- | :--- |
| **Python 3.x** | Linguagem de Desenvolvimento |
| **OpenCV (`cv2`)** | Segmentação, Detecção de Contornos, Medição Dimensional |
| **TensorFlow / Keras** | Implementação, Treinamento e Inferência da CNN |
| **NumPy** | Manipulação eficiente de dados de imagem |

## 📂 Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| `main.py` | **Orquestrador do Sistema.** Executa a sequência de inspeção (Dimensional → Visual) e imprime o veredito final no terminal, utilizando exemplos de teste definidos. |
| `inspecao_dimensional.py` | Lógica de OpenCV: binarização (com *fallback* para Otsu), detecção do contorno principal (`area > 5000`) e medição das dimensões. |
| `modelo_cnn.py` | Define a estrutura da CNN e implementa a função de inferência (`inspecao_visual_cnn`) para carregar e usar o modelo treinado. |
| `treinar_cnn.py` | Script para o treinamento e salvamento do modelo. Utiliza **Data Augmentation** para aumentar a robustez da CNN a variações de imagem. |
| `modelo_inspecao_visual.h5` | **Modelo CNN treinado.** Contém os pesos e a arquitetura prontos para a inspeção. |
| `dataset/` | Diretórios (simulados) para armazenamento das imagens de treinamento e validação. |
| `TRABALHO - Computer Vision.docx.pdf` | Documento com os requisitos teóricos e práticos do projeto. |

## 🚀 Como Iniciar e Executar

### 1. Pré-requisitos

Certifique-se de ter o Python 3.x instalado em seu ambiente.

### 2. Instalação de Dependências

Instale as bibliotecas necessárias para rodar o projeto:

```bash
pip install tensorflow opencv-python numpy