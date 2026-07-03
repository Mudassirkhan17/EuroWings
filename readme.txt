Please review final.ipynb for the full work.

The testing folders ipynb/ folder has older research notebooks (pred2, pred3, …). They are messy and optional — not needed to grade the main solution.

---

Streamlit demo (competitor price forecast)
------------------------------------------
1. Main notebook: final.ipynb
2. App entry point: streamlit_app.py
3. On deploy, the app trains from skyscanner_airfare_data.csv on first load (~1–2 min).
   Optional local export: python3 train_and_export.py → models/ (gitignored, too large for GitHub)

Run locally:
  pip install -r requirements.txt
  # optional: OPENAI_API_KEY in .env for AI insights after Predict
  streamlit run streamlit_app.py

Deploy on Streamlit Cloud (free):
  - Push this repo to GitHub (main branch)
  - https://share.streamlit.io → New app
  - Repo: Mudassirkhan17/EuroWings
  - Main file: streamlit_app.py
  - Python 3.10+
  - Add secret OPENAI_API_KEY in app Settings → Secrets (for AI summary)
