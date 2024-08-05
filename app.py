import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

st.title("Modelo de Rede Neural com Streamlit")

# Carregar e visualizar os dados
st.subheader("Carregar e Visualizar Dados")
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Primeiras linhas do dataset:")
    st.write(df.head())

    # Separar variáveis independentes e dependentes
    y = df['Chance of Admit ']
    x = df.drop('Chance of Admit ', axis=1)

    x_treino, x_teste = x[0:300], x[300:]
    y_treino, y_teste = y[0:300], y[300:]

    st.write("Forma dos dados de treino:", x_treino.shape)

    # Criar e treinar o modelo
    st.subheader("Criar e Treinar o Modelo")

    if st.button("Treinar Modelo"):
        modelo = Sequential()
        modelo.add(Dense(units=3, activation='relu', input_dim=x.shape[1]))
        modelo.add(Dense(units=1, activation='linear'))

        modelo.compile(loss='mse', optimizer='adam', metrics=['mae'])
        resultado = modelo.fit(x_treino, y_treino, epochs=200, batch_size=32, validation_data=(x_teste, y_teste), verbose=0)

        st.write("Modelo treinado com sucesso!")

        # Plotar gráfico do histórico de treinamento
        st.subheader("Histórico de Treinamento")

        fig, ax = plt.subplots()
        ax.plot(resultado.history['loss'], label='Erro treino')
        ax.plot(resultado.history['val_loss'], label='Erro teste')
        ax.set_title('Histórico de Treinamento')
        ax.set_ylabel('Função de custo')
        ax.set_xlabel('Épocas de treinamento')
        ax.legend()

        # Exibir o gráfico no Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf, caption="Gráfico do Histórico de Treinamento")
        plt.close(fig)
