# MovieIQ — Final Submission

This is the merged and regenerated MovieIQ assessment project.

## Main files
- `MovieIQ.ipynb` — complete analysis for Stages 0–4.
- `movieiq_analysis.py` — reproducible analysis script.
- `MovieIQ.py` — Streamlit dashboard.
- `WRITTEN_ANSWERS.md` — concise written answers with verified results.
- `movies.csv` — supplied source dataset.
- `movies_cleaned.csv` — prepared dataset.
- `assets/` — project charts.
- `requirements.txt` — dependencies.
- `results.json` — verified results.

## Success definition
`success = 1` when `revenue > budget`, otherwise `0`.

## Model features
`budget`, `popularity`, `runtime`, `vote_average`.

Revenue is excluded from predictors to avoid target leakage.

## Run
```bash
pip install -r requirements.txt
streamlit run MovieIQ.py
```
