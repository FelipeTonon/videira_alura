def previsao(modelo, image):
    if modelo is None:
        st.error("Modelo não carregado. Não é possível fazer a previsão.")
        return
    
    # Make prediction
    output_data = modelo.predict(image)
    
    # Apply softmax if needed
    if output_data.shape[1] > 1:
        output_data = tf.nn.softmax(output_data[0]).numpy()
    else:
        output_data = [1 - output_data[0][0], output_data[0][0]]  # Binary classification adjustment
    
    # Define classes and create visualization
    classes = ['Immature', 'Mature']
    df = pd.DataFrame({
        'classes': classes,
        'probabilidades (%)': 100 * np.array(output_data)
    })
    
    # Create bar chart with Plotly
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de Classes de Catarata')
    
    st.plotly_chart(fig)
