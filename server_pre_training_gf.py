import os
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuração para evitar uso de GPU (caso necessário)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Caminho para o arquivo CSV
CSV_PATH = "/final_dataset.csv"

def load_labeled_data(csv_path):
    """Carrega os dados rotulados de um arquivo CSV."""
    chunk_size = 100000  # Número de linhas lidas por vez
    x_data, y_data = [], []

    # Criar LabelEncoder para os rótulos
    label_encoder = LabelEncoder()

    for chunk in pd.read_csv(csv_path, sep=",", engine="python", low_memory=True, chunksize=chunk_size):
        # Remove valores NaN
        #chunk = chunk.dropna()

        # Selecionar apenas colunas numéricas
        numerical_cols = chunk.select_dtypes(include=np.number).columns

        # Se não houver colunas numéricas suficientes, pular chunk
        if len(numerical_cols) < 2:
            continue

        # Separar características (features) e rótulos (labels)
        x_chunk = chunk[numerical_cols[:-1]].values.astype(np.float32)
        y_chunk = label_encoder.fit_transform(chunk.iloc[:, -1].values)

        x_data.append(x_chunk)
        y_data.append(y_chunk)

    # Concatenar os dados
    if x_data and y_data:  # Verifica se há dados antes de concatenar
        x_final = np.vstack(x_data)
        y_final = np.hstack(y_data)
    else:
        raise ValueError("Erro: Nenhum dado válido encontrado no CSV.")

    # Dividir em treino e validação
    return train_test_split(x_final, y_final, test_size=0.2, train_size=0.8, random_state=42)

def build_model(input_shape):
    """Define a arquitetura do modelo."""
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPool1D(pool_size=2, strides=2, padding='same'),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def pretrain_model(model, x_train, y_train, x_val, y_val):
    """Executa o pré-treinamento com os dados rotulados."""
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_val, y_val))
    model.save('models/pretrained_model.keras')
    print("Modelo pré-treinado salvo!")

def main():
    print("Dispositivos disponíveis:", tf.config.list_physical_devices('GPU'))

    # Carregar dados rotulados do CSV
    x_train, x_val, y_train, y_val = load_labeled_data(CSV_PATH)

    # Ajustar formato dos dados para Conv1D
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    # Criar e treinar o modelo
    model = build_model(x_train.shape[1])
    pretrain_model(model, x_train, y_train, x_val, y_val)

if __name__ == "__main__":
    main()