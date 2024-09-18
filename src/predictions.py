from src.preprocess import PreprocessInfo
from config import NAME_COMPANY, PATH_DATA_RAW, PATH_DATA_PROCESSED

# Definir las rutas de entrada y salida
input_path = PATH_DATA_RAW
output_path = PATH_DATA_PROCESSED

# Instanciar la clase y ejecutar el proceso de preprocesamiento
preprocessor = PreprocessInfo(input_path=input_path, output_path=output_path)
preprocessor.preprocess()

print(f"Dataset procesado y guardado en {output_path}")
