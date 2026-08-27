from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import mlflow
from mlflow import MlflowClient
import joblib
import os

############## CONFIGURACIÓN

BASE = Path(__file__).parent

DB_PATH = str(BASE / "databases" / "trueml_database.db")
TABLE_NAME = "database_challenge"

OUTPUT_DB = str(BASE / "databases" / "trueml_predictions.db")
OUTPUT_TABLE = "predictions"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(BASE.parent / 'mlflow.db').as_posix()}"
)
MODEL_NAME = "trueml_collections_logr"
ALIAS = "champion_log"
WOE_PATH = str(BASE / "woe_encoder.pkl")
IMAGE_PATH = str(BASE / "trueml_logo.webp")

# Las 14 columnas que vio el WOEEncoder al entrenarse
ENCODER_COLS = [
    'latest_communication_channel', 'minimum_payment', 'previous_payment_amount',
    'product', 'last_reminder_sent_days', 'debt_to_income_ratio',
    'opened_last_communication', 'used_chat_feature', 'count_comms_sent_last_30d',
    'homeowner', 'ever_missed_payment', 'age_of_debt_yrs', 'total_balance',
    'latest_communication_dow',
]

# Las 10 que usa el modelo
variables = [
    'previous_payment_amount', 'product', 'last_reminder_sent_days',
    'debt_to_income_ratio', 'opened_last_communication',
    'count_comms_sent_last_30d', 'ever_missed_payment', 'age_of_debt_yrs',
    'total_balance', 'latest_communication_dow',
]

display_cols = ["account_id"] + variables

st.set_page_config(page_title="TrueML Collections", layout="wide")


################ CARGA

def load_encoder():
    return joblib.load(WOE_PATH)


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    tablas = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    if TABLE_NAME not in tablas["name"].values:
        conn.close()
        raise ValueError(
            f"La tabla '{TABLE_NAME}' no existe en {DB_PATH}. "
            f"Tablas: {list(tablas['name'])}"
        )
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    thr = float(mv.tags["decision_threshold"])

    return model, thr, mv.version


def save_predictions(df_out):
    """Escribe la tabla de predicciones en la base de salida."""
    with sqlite3.connect(OUTPUT_DB) as conn:
        df_out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        conn.commit()


# ============================================================
# STREAMLIT
# ============================================================

st.image(IMAGE_PATH, use_container_width=True)
st.title("TrueML Challenge: Debt Prediction")
st.write("Demo of an app for the TrueML Challenge")

# ------------------------------------------------------------
# Datos y modelo
# ------------------------------------------------------------

df_raw = load_data()

encoder = load_encoder()
df = encoder.transform(df_raw[ENCODER_COLS])[variables]

try:
    model, threshold, version = load_model()
    st.success(
        f"Champion model v{version} loaded successfully. "
        f"Decision threshold: {threshold:.3f}"
    )
except Exception as e:
    st.error(f"Could not load the champion model: {e}")
    st.exception(e)
    st.stop()

st.subheader("Database")
st.dataframe(df_raw[display_cols], use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Predicción de todo el dataset
# ------------------------------------------------------------

st.subheader("Batch prediction")

proba = model.predict_proba(df)[:, 1]
pred = np.where(proba >= threshold, model.classes_[1], model.classes_[0])

df_out = df_raw[variables].copy()
if "account_id" in df_raw.columns:
    df_out.insert(0, "account_id", df_raw["account_id"])
df_out["probability"] = proba
df_out["prediction"] = pred
df_out["model_version"] = version
df_out["threshold"] = threshold
df_out["scored_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

c1, c2, c3 = st.columns(3)
c1.metric("Records scored", f"{len(df_out):,}")
c2.metric("Predicted positive", f"{(df_out['prediction'] == 1).sum():,}")
c3.metric("Mean probability", f"{proba.mean():.1%}")

st.dataframe(
    df_out.sort_values("probability", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="%.3f"
        ),
    },
)

col_a, col_b = st.columns(2)

with col_a:
    st.download_button(
        "Download as CSV",
        df_out.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
        use_container_width=True,
    )

with col_b:
    if st.button("Save to database", type="primary", use_container_width=True):
        try:
            save_predictions(df_out)
            st.success(
                f"{len(df_out):,} rows written to "
                f"`{Path(OUTPUT_DB).name}` / `{OUTPUT_TABLE}`."
            )
        except Exception as e:
            st.error(f"Could not write to the database: {e}")
            st.exception(e)


# ------------------------------------------------------------
# Predicción individual
# ------------------------------------------------------------
st.subheader("Single prediction")

account_id = st.selectbox(
    "Select account_id",
    df_raw["account_id"].tolist(),
    help="Cuenta a evaluar con el modelo champion.",
)

# Posición de esa cuenta dentro del array de probabilidades
pos = int(np.flatnonzero(df_raw["account_id"].values == account_id)[0])

st.write("Selected record:")
st.dataframe(df_raw[display_cols].iloc[[pos]], hide_index=True)

if st.button("Predict"):
    probability = float(proba[pos])
    prediction = int(probability >= threshold)

    st.subheader("Prediction")

    col1, col2 = st.columns(2)
    col1.metric("Probability", f"{probability:.2%}")
    col2.metric("Threshold", f"{threshold:.2%}")

    if prediction == 1:
        st.success(f"Account {account_id}: POSITIVE (pays on time)")
    else:
        st.error(f"Account {account_id}: NEGATIVE (at risk)")