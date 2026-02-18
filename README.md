# FinTech Benchmark Tool

A Streamlit web app that automates structured competitor analysis using Gemini + Google Search grounding.
Converted from the original Google Colab notebook.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How to use

1. **Sidebar** – paste your Gemini API key (get one free at https://aistudio.google.com/app/apikey)

2. **① Companies** – enter one company name per line. The tool ships with 10 defaults; replace them freely.

3. **② Criteria** – edit, add, or delete scoring dimensions. For each criterion you set:
   - Category & name
   - Description (shown verbatim to the model)
   - Scale: 3, 4, or 5 points
   - Low anchor text (what score 1 looks like)
   - High anchor text (what the top score looks like)

4. **③ Few-shot Examples** – for any criterion, add real companies you've already scored. The model uses these as calibration anchors, which significantly improves consistency.

5. **④ Run Analysis** – preview the generated prompt, then click **Start Benchmark Run**. Progress is shown live. When done, download results as CSV or Excel.

## Output columns (per criterion)

| Column | Content |
|--------|---------|
| `Category › Name \| Score` | Numeric score (1–N) |
| `Category › Name \| Begründung` | Model's reasoning |
| `Category › Name \| Quellen` | Source URLs used |
| `Hinweise Datenlage` | Data gaps / caveats |

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → New app → select your repo
3. Users enter their own API key in the sidebar — no secrets needed in the repo

## Notes

- Uses `gemini-2.0-flash` with Google Search grounding (same model as the original notebook)
- Rate-limited to 1 request per 3 seconds to avoid quota errors
- All config (companies, criteria, examples) lives in session state — it resets on page refresh. For persistent config, export the JSON via the browser's developer console or extend the app with `st.download_button` on the session state.
