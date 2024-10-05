from src.preprocess import PreprocessInfo
from config import config

# Definir las rutas de entrada y salida
input_path = config.PATH_DATA_RAW
output_path = config.PATH_DATA_PROCESSED
print(input)

# Instanciar la clase y ejecutar el proceso de preprocesamiento
preprocessor = PreprocessInfo(input_path, output_path)
try:
    print(f"==== Ejecutando transformaciones de datos:")
    preprocessor.preprocess()
    print(f"==== Dataset procesado y guardado correctamente en {output_path}")
except FileNotFoundError:
    raise Exception(f"Error: No se encontró el archivo en la ruta {input_path}")
    