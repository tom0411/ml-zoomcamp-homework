# Heart disease screening

Deployed on AWS Elastic Beanstalk in `ap-southeast-1`:

http://heart-disease-mlz.ap-southeast-1.elasticbeanstalk.com/

A clinic collects a short set of measurements at a visit: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, oldpeak, and ST slope. This project estimates the probability that the patient has heart disease.

The score is a screening aid. A probability of 0.5 or higher means "refer for follow-up." A doctor still makes the diagnosis. The service is a form at `GET /` and a JSON API at `POST /predict`.

## Dataset

`heart.csv` is in this folder (918 rows, 12 columns), the [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The target is `HeartDisease`. There are no blank cells. A value of 0 in `RestingBP` or `Cholesterol` is treated as missing; 0 in `Oldpeak`, `FastingBS`, and `HeartDisease` is a real measurement.

## EDA and model

The data is split 60/20/20 with `random_state=1`. Medians for the two missing columns are computed on the training rows only, after the split. Categorical fields are one-hot encoded with `DictVectorizer`.

Risk ratios on the train+validation rows single out exercise-induced angina, a flat ST slope, asymptomatic chest pain, and fasting blood sugar above 120 mg/dl.

| Model | Validation ROC AUC | Notes |
| --- | --- | --- |
| Logistic regression (`lbfgs`) | 0.902 | Train AUC 0.926 |
| Random forest | 0.909 | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=1` |

The chosen forest is refit on train+validation and scored once on the held-out test set: **test ROC AUC 0.940**. `min_samples_leaf=1` memorizes training rows (train AUC 1.000). The validation score still beats logistic regression, so that model is the one served. Filling missing cholesterol with the median also hides a strong signal: rows with unmeasured cholesterol had a much higher disease rate.

## Run locally

Python 3.10. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py
```

`train.py` prints the test AUC and writes `model.bin` (medians, `DictVectorizer`, and the forest). `predict.py` loads that file and serves http://localhost:9696/.

## Docker

The image serves the already trained `model.bin`. It does not retrain.

```bash
docker build -t heart-disease .
docker run --rm -p 9696:9696 heart-disease
```

Open http://localhost:9696/.

## API

`POST /predict` takes one patient. Text fields are lowercased. A resting blood pressure or cholesterol of 0 is replaced with the median saved in `model.bin`.

```bash
curl -X POST http://heart-disease-mlz.ap-southeast-1.elasticbeanstalk.com/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 58,
    "sex": "M",
    "chestpaintype": "ASY",
    "restingbp": 140,
    "cholesterol": 0,
    "fastingbs": 1,
    "restingecg": "ST",
    "maxhr": 122,
    "exerciseangina": "Y",
    "oldpeak": 1.0,
    "st_slope": "Flat"
  }'
```

```json
{"heart_disease_probability": 0.991, "heart_disease": true}
```

Allowed categories: sex `M`/`F`; chest pain `ASY`/`ATA`/`NAP`/`TA`; fasting blood sugar `0`/`1`; resting ECG `Normal`/`ST`/`LVH`; exercise angina `Y`/`N`; ST slope `Flat`/`Up`/`Down`.
