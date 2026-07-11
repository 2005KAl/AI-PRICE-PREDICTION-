# Experiments UI

This is an isolated Flask-based UI to preview the experimental multi-model pipeline. It intentionally does not modify the main Flask app.

Prerequisites

- Python 3.10+
- Install Flask in your environment:

```bash
pip install flask pandas
```

Run

```bash
cd experiments/ui
python app.py
```

Open http://localhost:5001 in your browser.

Notes

- The UI selects the nearest sample row from the experimental dataset matching the chosen home type and uses that full row for prediction via the experiment `prediction_router`.
- This keeps the UI isolated and avoids reimplementing the full preprocessing and distance calculations in the frontend.
