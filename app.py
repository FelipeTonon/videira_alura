import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

@st.cache_resource
def carrega_modelo():
    # URL do modelo Keras (no formato .h5)
    # url = 'C:\Users\felipe.tonon\Desktop\Reconhecimento_Imagens\modelo_catarata.h5
    
    # Baixar o modelo
    output = 'C:\Users\felipe.tonon\Desktop\Reconhecimento_Imagens\modelo_catarata.h5'
    #gdown.download(url, output, quiet=False)
    
    # Carregar o modelo Keras normal
    modelo = tf.keras.models.load_model(output)
    
    return modelo

def carrega_imagem():
    # Cria um file uploader que permite o usuário carregar imagens
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Para ler a imagem como um objeto PIL Image
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Mostrar a imagem carregada
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        # Pré-processamento da imagem
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)

        return image

def previsao(modelo, image):
    # Faz a previsão
    output_data = modelo.predict(image)
    
    # Definir as classes e criar a visualização
    classes = ['Immature', 'Mature']
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    
    # Criar gráfico de barras com Plotly
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de Classes de Catarata')
    
    st.plotly_chart(fig)

def main():
    # Detalhes da página inicial
    st.set_page_config(
        page_title="Classificador de Catarata",
        page_icon="👁️",
    )
    st.write("# Classificador de Catarata! 👁️")

    # Carrega o modelo
    modelo = carrega_modelo()
    
    # Carrega a imagem
    image = carrega_imagem()

    # Classifica a imagem se ela for carregada
    if image is not None:
        previsao(modelo, image)

if __name__ == "__main__":
    main()
