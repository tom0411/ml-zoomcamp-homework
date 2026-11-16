import pickle
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_FILE = Path(__file__).parent / "model.bin"
INDEX_FILE = Path(__file__).parent / "static" / "index.html"

with MODEL_FILE.open("rb") as f_in:
    medians, dv, model = pickle.load(f_in)

app = FastAPI(title="Heart disease prediction")

TEXT_FIELDS = ["sex", "chestpaintype", "restingecg", "exerciseangina", "st_slope"]


class Patient(BaseModel):
    age: float = Field(examples=[49])
    sex: str = Field(examples=["M"])
    chestpaintype: str = Field(examples=["ASY"])
    restingbp: float = Field(examples=[140])
    cholesterol: float = Field(examples=[234])
    fastingbs: int = Field(examples=[0])
    restingecg: str = Field(examples=["Normal"])
    maxhr: float = Field(examples=[140])
    exerciseangina: str = Field(examples=["Y"])
    oldpeak: float = Field(examples=[1.0])
    st_slope: str = Field(examples=["Flat"])


class Prediction(BaseModel):
    heart_disease_probability: float
    heart_disease: bool


def prepare(patient: Patient):
    record = patient.model_dump()
    for field in TEXT_FIELDS:
        record[field] = record[field].strip().lower().replace(" ", "_")
    for field in ["restingbp", "cholesterol"]:
        if record[field] == 0:
            record[field] = medians[field]
    return record


@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    X = dv.transform([prepare(patient)])
    probability = float(model.predict_proba(X)[0, 1])
    return Prediction(
        heart_disease_probability=probability,
        heart_disease=probability >= 0.5,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9696)
