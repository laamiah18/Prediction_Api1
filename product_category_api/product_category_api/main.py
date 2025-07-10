from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("model/product_category.pkl", "rb") as f:
    model, scaler, le, cluster_map, model_name = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": None})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, age: int = Form(...), gender: str = Form(...)):
    gender = gender.capitalize()
    gender_encoded = le.transform([gender])[0]

    input_df = pd.DataFrame([[age, gender_encoded]], columns=["Customer Age", "Gender"])
    features = scaler.transform(input_df)
    cluster = model.predict(features)[0]
    recommendations = cluster_map.get(cluster, ["No suggestions available"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "recommendations": recommendations,
        "age": age,
        "gender": gender
    })
