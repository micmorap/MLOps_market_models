# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from io import StringIO
from src.preprocess import PreprocessInfo
from src.prediction_close import StockPricePredictor
from pydantic import BaseModel
import databases
import sqlalchemy
from contextlib import asynccontextmanager
from sqlalchemy import func, select
from datetime import datetime, date

# 1. Create and instiate the Database:
DATABASE_URL = 'sqlite:///./save_predictions.db'
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

items = sqlalchemy.Table(
    "historical_predictions",
    metadata,
    sqlalchemy.Column("id_record", sqlalchemy.Integer, primary_key = True),
    sqlalchemy.Column("prediction_value", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("date_prediction", sqlalchemy.DateTime, nullable=False)
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# Pydantic model for validation
class HistoricalPredictions(BaseModel):
    id_record: int
    prediction_value: float
    date_prediction: date



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

# 5. Endpoint to get stock price prediction:
@app.get('/get_stock_price_prediction')
async def get_price_prediction():
    try:
        predicted_value = StockPricePredictor(model_path=MODEL_PATH, data_path=PATH_DATA_PROCESSED).run()

        # Inserta el valor predicho en la base de datos
        query = items.insert().values(
            prediction_value = predicted_value,
            date_prediction=datetime.now()  # Obtiene la fecha y hora actual
        )

        record_id = await database.execute(query) 
    
    except Exception as e:
            raise HTTPException(status_code=404, detail=f"Error calculating stock price prediction for {NAME_COMPANY}. {str(e)}")
        

    return {"id_record": record_id, "predicted_stock_price": predicted_value}
