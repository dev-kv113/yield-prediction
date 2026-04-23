import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Sugarcane Yield Prediction", layout="wide")

st.header("AI Based Advanced Sugarcane Yield Prediction Model")
st.write(
    "Proprietary AI based yield perdition model developed by Geotrans Technologies Pvt. Ltd."
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Try to derive useful date features from any datetime-like columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                if dt.notna().sum() > 0:
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_day"] = dt.dt.day
                    df[f"{col}_dayofyear"] = dt.dt.dayofyear
                    df = df.drop(columns=[col])
            except Exception:
                pass

    return df


def train_model(df: pd.DataFrame, target_col: str):
    df = build_features(df)
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }

    return pipeline, metrics, X.columns.tolist()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("1. Upload training data")
training_file = st.sidebar.file_uploader("Upload CSV training file", type=["csv"])

if training_file is None:
    st.info("Please upload a CSV file containing historical sugarcane data.")
    st.stop()

try:
    df = load_csv(training_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("Training data preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Columns detected")
st.write(list(df.columns))

target_col = st.selectbox(
    "Select the target column (the yield column to predict)",
    options=df.columns.tolist(),
)

if st.button("Train model"):
    try:
        model_pipeline, metrics, feature_columns = train_model(df, target_col)
        st.session_state["model_pipeline"] = model_pipeline
        st.session_state["feature_columns"] = feature_columns
        st.session_state["target_col"] = target_col
        st.session_state["trained_df_columns"] = df.columns.tolist()

        st.success("Model trained successfully.")

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['MAE']:.3f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
        c3.metric("R²", f"{metrics['R2']:.3f}")

        model_bytes = io.BytesIO()
        joblib.dump(model_pipeline, model_bytes)
        st.download_button(
            label="Download trained model",
            data=model_bytes.getvalue(),
            file_name="sugarcane_yield_model.joblib",
            mime="application/octet-stream",
        )
    except Exception as e:
        st.error(f"Training failed: {e}")

if "model_pipeline" in st.session_state:
    st.subheader("Predict on new data")
    st.write(
        "Upload a CSV with the same predictor columns as the training data, excluding the target column."
    )

    predict_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict")

    if predict_file is not None:
        try:
            new_df = pd.read_csv(predict_file)
            st.write("Prediction input preview")
            st.dataframe(new_df.head(), use_container_width=True)

            new_df_processed = build_features(new_df)
            predictions = st.session_state["model_pipeline"].predict(new_df_processed)
            result_df = new_df.copy()
            result_df[f"predicted_{st.session_state['target_col']}"] = predictions

            st.subheader("Prediction results")
            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_out,
                file_name="sugarcane_yield_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Train the model first to enable prediction.")
