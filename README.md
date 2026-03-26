# MarketBasket Engine

A Flask-based market basket analysis dashboard that combines **Apriori association rules** and **collaborative filtering** to deliver product and customer recommendations.

---

## Project Structure

```
marketbasket/
│
├── app.py                        ← Flask app (routes + utility functions)
│
├── model/
│   ├── __init__.py
│   ├── apriori_model.py          ← Association rule mining (Apriori)
│   ├── collaborative_filtering.py← User-based CF recommendations
│   └── lstm_prediction.py        ← Optional sequence predictor (TensorFlow)
│
├── templates/
│   └── index.html                ← Main dashboard (Jinja2 template)
│
├── static/
│   ├── style.css                 ← All styles (CSS variables + components)
│   └── charts.js                 ← Chart.js helpers (item + rules charts)
│
└── data/
    └── default.csv               ← Default dataset (upload to replace)
```

---

## Quick Start

```bash
pip install flask pandas scikit-learn werkzeug
python app.py
# Open http://127.0.0.1:5000
```

### Optional: LSTM predictions
```bash
pip install tensorflow
```

---

## CSV Format

Your dataset needs at minimum:

| Column   | Example          | Notes                          |
|----------|------------------|--------------------------------|
| Items    | Milk, Bread, Eggs | Comma-separated item names     |
| UserID   | 101               | Required for CF recommendations |

Column names are case-insensitive.

---

## Features

| Feature | Model | Route |
|---------|-------|-------|
| Product associations | Apriori | POST / |
| Bundle detection | Apriori | GET / |
| Personalised recs | Collaborative Filtering | POST / |
| Popularity fallback | CF | GET / |
| Sequence prediction | LSTM (optional) | — |
| Executive report | — | GET /download_report |
| Dataset upload | — | POST /upload |