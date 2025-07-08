from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("model/best_clustering_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, age: int = Form(...), gender: str = Form(...), aov: float = Form(...)):
    gender_encoded = 0 if gender.lower() == "female" else 1
    input_data = np.array([[age, gender_encoded, aov]])
    scaled_input = scaler.transform(input_data)
    cluster = int(model.predict(scaled_input)[0])
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": f"Predicted Cluster: {cluster}"
    })
