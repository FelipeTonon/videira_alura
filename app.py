import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import gdown
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    url = 'https://drive.google.com/uc?id=1AYuryrifPeC6BNcwYnRQmjKc37mkJXNX'
    gdown.download(url, 'modelo_binario.tflite', quiet=False)
    interpreter = tf.lite.Interpreter(model_path='modelo_binario.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Mostrar a imagem carregada
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        # Pré-processamento da imagem
        image = image.resize((224, 224))  # Ajuste para o tamanho de entrada do seu modelo
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):
    # Obtém detalhes dos tensores de entrada e saída
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define o tensor de entrada para o modelo
    interpreter.set_tensor(input_details[0]['index'], image)

    # Executa a inferência
    interpreter.invoke()

    # Obtém a saída do modelo
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Verifica o formato da saída e processa adequadamente
    if output_data.shape == (1, 1):
        probabilidade = output_data[0][0]
        classes = ['Saudável', 'Doente']
        probabilidades = [100 * (1 - probabilidade), 100 * probabilidade]
    elif output_data.shape == (1, 2):
        probabilidades = 100 * output_data[0]
        classes = ['Saudável', 'Doente']
    else:
        st.error("Formato de saída do modelo não suportado.")
        return

    df = pd.DataFrame({
        'classes': classes,
        'probabilidades (%)': probabilidades
    })
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de Classes')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videira",
        page_icon="🍇",
    )
    
    st.write("# Classifica Folhas de Videira! 🍇")
    
    interpreter = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()
