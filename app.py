import streamlit as st
import pandas as pd
import joblib
import io

FEATURES = ['pl_pnum','ra','dec','pl_tranmid','pl_trandurh','pl_trandep','st_tmag','st_tmagerr1','st_tmagerr2']
model = joblib.load("best_model_Gradient_Boosting (3).pkl")

st.title("Hunting Exoplanets with AI")

with st.expander("Download template"):
    buf = io.StringIO()
    pd.DataFrame(columns=FEATURES).to_csv(buf, index=False)
    st.download_button("Download template CSV", buf.getvalue().encode("utf-8"), "template.csv", "text/csv")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    missing = [c for c in FEATURES if c not in df_raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    X = df_raw[FEATURES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = df_raw.copy()
    out["prediction"] = pred
    out["probability"] = proba
    st.dataframe(out.head())
    st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")



