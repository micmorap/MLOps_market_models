# Declare general variables
# To productive models:
NAME_COMPANY = "ISA_Historical_Info"
PATH_DATA_RAW = f"../data/raw/{NAME_COMPANY}.csv"
PATH_DATA_PROCESSED = f"../data/processed/{NAME_COMPANY}.csv"
MODEL_PATH = f"../models/model_lstm_{NAME_COMPANY}.h5"

# To training models:
PATH_FINAL_TRAIN = f"../data/processed/processed_training_set_{NAME_COMPANY}"
MIN_MAX_SCALER_TRANSFORMER_PATH =  "../config/model/transformers_min_max_scaler.pkl"
PATH_FINAL_TEST = f"../data/processed/processed_testing_set_{NAME_COMPANY}"
PATH_FINAL_X_TRAIN = f"../data/final/final_X_training_set_{NAME_COMPANY}"
PATH_FINAL_Y_TRAIN = f"../data/final/final_Y_training_set_{NAME_COMPANY}"
PATH_FINAL_X_TEST = f"../data/final/final_X_training_set_{NAME_COMPANY}"
PATH_FINAL_Y_TEST = f"../data/final/final_Y_training_set_{NAME_COMPANY}"

