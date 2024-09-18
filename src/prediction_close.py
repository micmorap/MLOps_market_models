import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from config import MODEL_PATH, PATH_DATA_PROCESSED

class StockPricePredictor:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.model = None

    def load_data(self):
        """Cargar el dataset procesado y filtrar la columna 'Close'."""
        self.dataset = pd.read_csv(self.data_path)
        self.dataset = self.dataset['Close'].tail(60).values  # Filtrar últimos 60 registros
        self.dataset = self.dataset.reshape(-1, 1)  # Convertir a array 2D

    def normalize_data(self):
        """Normalizar los datos entre 0 y 1."""
        self.dataset = self.scaler.fit_transform(self.dataset)

    def prepare_input(self):
        """Preparar los datos para que puedan ser usados como input en el modelo."""
        return np.array([self.dataset])

    def load_model(self):
        """Cargar el modelo guardado en formato .h5."""
        self.model = load_model(self.model_path)

    def predict(self):
        """Realizar la predicción y desnormalizar el valor."""
        normalized_prediction = self.model.predict(self.prepare_input())
        prediction = self.scaler.inverse_transform(normalized_prediction)
        return prediction[0][0]  # Devuelve el valor desnormalizado

    def run(self):
        """Ejecutar todo el pipeline: cargar datos, normalizar, predecir, y desnormalizar."""
        self.load_data()
        self.normalize_data()
        self.load_model()
        prediction = self.predict()
        print(f"Predicción desnormalizada del valor de la acción: {prediction:.2f}")

# Ejemplo de uso
if __name__ == "__main__":
    model_path = MODEL_PATH  # Ruta al archivo del modelo .h5
    data_path = PATH_DATA_PROCESSED  # Ruta al archivo de datos procesados

    predictor = StockPricePredictor(model_path, data_path)
    predictor.run()
