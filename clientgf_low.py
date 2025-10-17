import argparse
import os
from pathlib import Path
import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import gc
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações para reduzir uso de memória
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Desabilita logs do TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)  # Limita threads da CPU
tf.config.threading.set_inter_op_parallelism_threads(1)

# Cliente Federado
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        """Retorna os parâmetros do modelo local."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Treina o modelo localmente e retorna os novos parâmetros."""
        self.model.set_weights(parameters)
        
        # Hiperparâmetros ajustados para economia de memória
        batch_size = config.get("batch_size", 8)  # Batch size reduzido
        epochs = config.get("local_epochs", 1)    # Apenas 1 época
        validation_split = config.get("validation_split", 0.2)

        # Treinamento com callbacks para evitar desperdício
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0,  # Desabilita logs de treinamento
        )

        # Libera memória após o treinamento
        del self.x_train, self.y_train
        gc.collect()

        return self.model.get_weights(), len(self.x_train), {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }

    def evaluate(self, parameters, config):
        """Avalia o modelo localmente."""
        self.model.set_weights(parameters)
        
        # Avaliação local
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_bool = np.argmax(y_pred, axis=-1)
        
        # Matriz de confusão e relatório de classificação
        confusion = confusion_matrix(self.y_test, y_pred_bool)
        # Criar um heatmap e salvar a imagem
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
    
        # Salvar a imagem
        plt.savefig("confusion_matrix.png")
        plt.close()  # Fecha a figura para economizar memória
    
        print("Matriz de Confusão salva como confusion_matrix.png")


        print('Matriz de Confusão:\n', confusion)
        print('\nRelatório de Classificação:\n', classification_report(self.y_test, y_pred_bool))

        # Calcula a perda e a acurácia
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Libera memória após a avaliação
        del self.x_test, self.y_test
        gc.collect()

        return loss, len(self.x_test), {"accuracy": accuracy}

# Função para carregar dados em partes
def load_data_in_chunks(filename, chunk_size=10000):
    """Carrega dados em partes para economizar memória."""
    chunks = pd.read_csv(filename, chunksize=chunk_size)
    data = []
    for chunk in chunks:
        chunk = preprocess_data(chunk)
        data.append(chunk)
    return np.vstack(data).astype(np.float32)

# Função para pré-processamento
def preprocess_data(df):
    """Pré-processa os dados."""
    df = df.apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x)
    return StandardScaler().fit_transform(df)

# Função principal
def main():
    parser = argparse.ArgumentParser(description="Cliente Federated Learning")
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--address", type=str, required=True)
    args = parser.parse_args()

    # Definição do modelo
    model = Sequential([
        Conv1D(16, 3, activation='relu', input_shape=(10, 1)),  # 10 características
        Conv1D(32, 3, activation='relu'),
        MaxPool1D(pool_size=2),  # Pooling mais agressivo
        Flatten(),
        Dense(16, activation='relu'),  # Menos neurônios
        Dense(5, activation='softmax'),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Carrega dados de treinamento em partes
    data = load_data_in_chunks("/home/nodeflower/client/client_1_data.csv", chunk_size=10000)
    x = data[:, :-1]  # Todas as colunas, exceto a última
    y = data[:, -1]   # Última coluna (rótulos)

    #redimensiona
    x =x[:, :10]

    # Divide os dados em treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Redimensiona para 10 características
    x_train = x_train.reshape(x_train.shape[0], 10, 1)
    x_test = x_test.reshape(x_test.shape[0], 10, 1)

    # Inicializa o cliente
    client = CifarClient(model, x_train, y_train, x_test, y_test).to_client()
    fl.client.start_client(server_address=args.address, client=client)

if __name__ == "__main__":
    main()