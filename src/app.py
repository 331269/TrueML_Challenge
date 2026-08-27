import os
import sqlite3
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow import MlflowClient


############## CONFIGURATION

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

# Columns the WOE encoder was fitted on
ENCODER_COLS = [
    'latest_communication_channel', 'minimum_payment', 'previous_payment_amount',
    'product', 'last_reminder_sent_days', 'debt_to_income_ratio',
    'opened_last_communication', 'used_chat_feature', 'count_comms_sent_last_30d',
    'homeowner', 'ever_missed_payment', 'age_of_debt_yrs', 'total_balance',
    'latest_communication_dow',
]

# Features selected by RFE, in training order
variables = [
    'previous_payment_amount', 'product', 'last_reminder_sent_days',
    'debt_to_income_ratio', 'opened_last_communication',
    'count_comms_sent_last_30d', 'ever_missed_payment', 'age_of_debt_yrs',
    'total_balance', 'latest_communication_dow',
]

display_cols = ["account_id"] + variables

st.set_page_config(page_title="TrueML Collections", layout="wide")


############## LOADERS

def load_encoder():
    return joblib.load(WOE_PATH)


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    if TABLE_NAME not in tables["name"].values:
        conn.close()
        raise ValueError(
            f"Table '{TABLE_NAME}' not found in {DB_PATH}. "
            f"Available tables: {list(tables['name'])}"
        )
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    # Resolve the artifact folder locally: the registry stores absolute
    # Windows paths that do not exist on the deployment host
    model_id = mv.source.rsplit("/", 1)[-1]
    matches = [p.parent for p in BASE.parent.rglob("MLmodel") if model_id in str(p)]
    if not matches:
        raise FileNotFoundError(
            f"No artifacts found for {model_id} under {BASE.parent}."
        )

    model = mlflow.sklearn.load_model(str(matches[0]))

    return model, float(mv.tags["decision_threshold"]), mv.version


def save_predictions(df_out):
    """Write the predictions table to the output database."""
    with sqlite3.connect(OUTPUT_DB) as conn:
        df_out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        conn.commit()


############## HEADER

st.image(IMAGE_PATH, use_container_width=True)
st.title("TrueML Challenge: Debt Prediction")
st.write("Demo of an app for the TrueML Challenge")


############## DATA AND MODEL

df_raw = load_data()

encoder = load_encoder()
df = encoder.transform(df_raw[ENCODER_COLS])[variables]

try:
    model, threshold, version = load_model()
    st.success(
        f"Champion model v{version} loaded successfully. "
    )
except Exception as e:
    st.error(f"Could not load the champion model: {e}")
    st.exception(e)
    st.stop()

st.subheader("Database")
st.dataframe(df_raw[display_cols], use_container_width=True, hide_index=True)


############## BATCH PREDICTION

st.subheader("Batch prediction")

proba = model.predict_proba(df)[:, 1]
pred = np.where(proba >= threshold, model.classes_[1], model.classes_[0])

df_out = df_raw[display_cols].copy()
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
        except Exception:
            st.warning(
                "Writing to the database is only available when running locally. "
                "Use the CSV download instead."
            )
