from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd

# Load the model
with open("model/regression_grid.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Mount static and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, age: int = Form(...), gender: str = Form(...)):
    gender = gender.lower()
    if gender not in ["male", "female"]:
        result = "Invalid gender. Choose 'Male' or 'Female'."
    else:
        gender_encoded = 1 if gender == "male" else 0
        df = pd.DataFrame([[age, gender_encoded]], columns=["Customer Age", "Gender"])
        prediction = model.predict(df)[0]
        result = f"Predicted Average Order Value: ₹{round(prediction, 2)}"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
