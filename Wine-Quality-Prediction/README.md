# Wine Quality Prediction Web App (PCA)

Simple end-to-end project to predict wine quality as Good Wine or Bad Wine using a CSV dataset and a web form.

## Project Goal

Build a small local web app where:

1. The app learns from past wine records in a CSV file.
2. A user enters new wine values in a browser form.
3. The app returns a quality prediction with confidence.

## How the Flow Works

1. App starts and reads `winequality.csv`.
2. It creates a binary target from `quality`:
	 - Good Wine: `quality >= 6`
	 - Bad Wine: `quality < 6`
3. It trains a pipeline (scaling, PCA, classification).
4. Browser opens the form page.
5. User submits wine feature values.
6. Backend predicts and returns result JSON.
7. Frontend shows final label and confidence.

## Main Files

- `app.py` - backend app, model training, and API routes.
- `winequality.csv` - training dataset.
- `templates/index.html` - form UI and result display.
- `static/styles.css` - page styling.
- `requirements.txt` - Python dependencies.
- `wine.py` - old exploratory script (not used by the web app).

## Folder Structure

```text
Wine-Quality-Prediction/
	app.py
	winequality.csv
	requirements.txt
	wine.py
	templates/
		index.html
	static/
		styles.css
```

## Setup and Run

### Option A: Run from project folder

```powershell
cd .\Wine-Quality-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### Option B: Run from workspace root

```powershell
d:/PCA-Principle-Component-Analysis-For-Wine-dataset/.venv/Scripts/python.exe -m pip install -r .\Wine-Quality-Prediction\requirements.txt
d:/PCA-Principle-Component-Analysis-For-Wine-dataset/.venv/Scripts/python.exe .\Wine-Quality-Prediction\app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## API

### GET /api/metrics

Returns model summary values (accuracy, rows, PCA components).

### POST /api/predict

Content type: `application/json`

Request body example:

```json
{
	"fixed_acidity": 7.4,
	"volatile_acidity": 0.7,
	"citric_acid": 0.0,
	"residual_sugar": 1.9,
	"chlorides": 0.076,
	"free_sulfur_dioxide": 11.0,
	"total_sulfur_dioxide": 34.0,
	"density": 0.9978,
	"pH": 3.51,
	"sulphates": 0.56,
	"alcohol": 9.4
}
```

Response example:

```json
{
	"ok": true,
	"label": "Bad Wine",
	"good_probability": 20.04,
	"rule": "Quality >= 6 is treated as Good Wine."
}
```

## Notes

1. The model is not pretrained. It is trained at startup from the CSV.
2. The web app is the current main workflow.
3. The app runs on Flask development server.

## Troubleshooting

1. `Could not open requirements file`
	 - Use the correct path to `Wine-Quality-Prediction/requirements.txt`.

2. `can't open file ... app.py`
	 - Run with the correct path:
		 `...python.exe .\Wine-Quality-Prediction\app.py`

3. Browser does not open
	 - Manually open `http://127.0.0.1:5000`.

4. Port already in use
	 - Stop older Python/Flask process and restart.


