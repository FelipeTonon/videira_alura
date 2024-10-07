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
    
    gdown.download(url,'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    
    return interpreter


def carrega_imagem():
    # Cria um file uploader que permite o usuário carregar imagens
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma:", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Para ler a imagem como um objeto PIL Image
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Mostrar a imagem carregada
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        #Pré-processamento da imagem
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

    # Verifica o formato e os dados do output
    st.write("Output data shape:", output_data.shape)
    st.write("Output data:", output_data)

    classes = ['Immature', 'Mature']

    if len(output_data[0]) == len(classes):
        df = pd.DataFrame()
        df['classes'] = classes
        df['probabilidades (%)'] = 100 * output_data[0]
        fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                     title='Probabilidade de nível de Catarata:')
        st.plotly_chart(fig)
    else:
        st.error("Erro: O número de saídas do modelo não corresponde ao número de classes.")

def main():
    st.set_page_config(
        page_title="Classifica Nível de Catarata! 👁️",
        page_icon="👁️",
    )
    
    st.write("# Classifica Nível de Catarata! 👁️")
    

    interpreter = carrega_modelo()

    image = carrega_imagem()

    if image is not None:

        previsao(interpreter,image)
    


if __name__ == "__main__":
    main()