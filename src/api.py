from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import pandas as pd

from utils import *

import data_pipeline as data_pipeline
import preprocessing as preprocessing

config = load_config()

model_data = pickle_load('../' + config['final_model']['model_directory'] + config['final_model']['model_name'])

class ApiData(BaseModel):
    age : int
    sex : str
    browser : str
    source : str
    purchase_value : int

app = FastAPI()

@app.get('/')
def home():
    return "Hello, E-commerce Transaction Fraud Detection API up!"

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080, reload=True)


@app.post("/predict/")
async def predict(data: ApiData):    

    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    
    # data defence
    try:
        data_pipeline.check_data(data, config)
    except AssertionError as ae:
        return {"result": [], "error_msg": str(ae)}

    # encoding
    data = ohe_input_new_data(data, config)
    data = pd.DataFrame.from_dict(data)

    # predict data
    y_pred = model_data.predict(data)

    # inverse transform
    label = ['Not Fraud', 'Fraud']
    y_pred = label[y_pred[0]]

    # return data
    return {"res" : y_pred, "error_msg": ""}
    
print('API OK')
