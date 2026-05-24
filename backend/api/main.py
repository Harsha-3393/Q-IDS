from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import pandas as pd
import joblib


app = FastAPI(
    title="Q-IDS API"
)

# Enable frontend requests
app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

model = joblib.load(
    "models/xgb_model.pkl"
)


class PredictionInput(BaseModel):
    dur: float
    state: float
    dpkts: float
    sbytes: float
    dbytes: float
    rate: float
    sttl: float
    dttl: float
    sload: float
    dload: float
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    tcprtt: float
    synack: float
    ackdat: float
    smean: float
    dmean: float
    ct_state_ttl: float


@app.post("/predict")
def predict(data: PredictionInput):

    values = pd.DataFrame(
        [data.dict()]
    )

    pred = model.predict(
        values
    )[0]

    return {
        "prediction":
        "attack" if pred == 1
        else "normal"
    }