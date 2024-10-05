import pandas as pd
import os

class PreprocessInfo:
    def __init__(self, input_path, output_path):
        self.input_path = input_path  # Ruta del dataset crudo
        self.output_path = output_path  # Ruta para guardar el dataset procesado
        self.dataset = None

    def load_data(self):
        """Cargar los datos desde el archivo CSV."""
        try:
            self.dataset = pd.read_csv(self.input_path)
        except FileNotFoundError:
            raise Exception(f"==== El archivo no fue encontrado en la ruta: {self.input_path}")

    def rename_columns(self):
        """Renombrar las columnas del dataset."""
        try:
            self.dataset.rename(columns={
                'Fecha': 'Date',
                'Último': 'Close',
                'Apertura': 'Open',
                'Máximo': 'High',
                'Mínimo': 'Low',
            }, inplace=True)
        except:
            raise Exception(f"Problemas asociados a los nombres de las columnas")

    def format_date(self):
        """Transformar el formato de la columna 'Date' de aaaa.mm.dd a aaaa-mm-dd."""
    
        try:
            self.dataset['Date'] = self.dataset['Date'].str.replace('.', '-', regex=False)
            self.dataset['Date'] = pd.to_datetime(self.dataset['Date'], format='%d-%m-%Y')
        except:
            raise Exception(f"Problemas en formato o estructura de la columna Date")

    @staticmethod
    def transformar_a_float(valor):
        """Transformar el valor de texto a float eliminando puntos y comas."""
        valor = valor.replace('.', '')  # Eliminar puntos de miles
        valor = valor.replace(',', '.')  # Reemplazar coma por punto decimal
        return float(valor)

    def convert_to_float(self):
        """Aplicar la transformación a las columnas numéricas."""
        columnas = ['Open', 'High', 'Low', 'Close']
        for columna in columnas:
            self.dataset[columna] = self.dataset[columna].apply(self.transformar_a_float)

    def save_processed_data(self):
        """Guardar el dataset procesado en la carpeta 'processed'."""
        self.dataset.to_csv(self.output_path, index=False)

    def preprocess(self):
        """Aplicar todas las transformaciones al dataset."""
        self.load_data()
        self.rename_columns()
        self.format_date()
        self.convert_to_float()
        self.save_processed_data()

