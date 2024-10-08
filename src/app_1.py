# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import keras
import os
from io import StringIO
from src.preprocess import PreprocessInfo
from src.prediction_close import StockPricePredictor


# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Create ENV variables:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Ruta al directorio del proyecto
NAME_COMPANY = "ISA_Historical_Info"
PATH_DATA_RAW = f"../data/raw/{NAME_COMPANY}.csv"
PATH_DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', f'{NAME_COMPANY}_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_lstm_ISA_Historical_Info.h5')

# 5. Endpoint to preprocess info:

@app.post('/preprocess_raw_info')
async def preprocess_raw_info(file: UploadFile = File(...)):    
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        df.to_csv(PATH_DATA_RAW)
        pre_processor = PreprocessInfo(PATH_DATA_RAW, PATH_DATA_PROCESSED)
        pre_processor.preprocess()
        logger = f"Cargue y transformación exitosos en {PATH_DATA_PROCESSED}"
    except:
        raise HTTPException(status_code=404, detail="Item not found")
    return logger

# 5. Create instances:
#model_predictor = StockPricePredictor(MODEL_PATH, PATH_DATA_PROCESSED)