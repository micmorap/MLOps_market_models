# Declare general variables
# To productive models:
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Ruta al directorio del proyecto
NAME_COMPANY = "ISA_Historical_Info"
PATH_DATA_RAW = f"data/raw/{NAME_COMPANY}.csv"
PATH_DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', f'{NAME_COMPANY}_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_lstm_ISA_Historical_Info.h5')

# To training models:
PATH_FINAL_TRAIN = f"../data/processed/processed_training_set_{NAME_COMPANY}"
MIN_MAX_SCALER_TRANSFORMER_PATH =  "../config/model/transformers_min_max_scaler.pkl"
PATH_FINAL_TEST = f"../data/processed/processed_testing_set_{NAME_COMPANY}"
PATH_FINAL_X_TRAIN = f"../data/final/final_X_training_set_{NAME_COMPANY}"
PATH_FINAL_Y_TRAIN = f"../data/final/final_Y_training_set_{NAME_COMPANY}"
PATH_FINAL_X_TEST = f"../data/final/final_X_training_set_{NAME_COMPANY}"
PATH_FINAL_Y_TEST = f"../data/final/final_Y_training_set_{NAME_COMPANY}"

