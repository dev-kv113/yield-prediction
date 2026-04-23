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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Sugarcane Yield Prediction", layout="wide")

st.image("logo.png", width=120)  # adjust size if needed
st.markdown(
    """
    <h2 style="margin-top: -20px;">
        AI Based Advanced Sugarcane Yield Prediction Model
    </h2>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Proprietary AI based yield prediction model developed by Geotrans Technologies Pvt. Ltd."
)

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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


def adjusted_r2_score(r2, n, p):
    if n > p + 1:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return np.nan


def get_preprocessor(X: pd.DataFrame):
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

    return preprocessor


def train_hybrid_model(df: pd.DataFrame, target_col: str, linear_weight=0.4, rf_weight=0.6):
    df = build_features(df)
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = get_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_processed, y_train)
    linear_test_pred = linear_model.predict(X_test_processed)

    # Pure Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_processed, y_train)
    rf_test_pred = rf_model.predict(X_test_processed)

    # Final Weighted Hybrid
    hybrid_test_pred = linear_weight * linear_test_pred + rf_weight * rf_test_pred

    # Metrics
    linear_mae = float(mean_absolute_error(y_test, linear_test_pred))
    linear_rmse = float(np.sqrt(mean_squared_error(y_test, linear_test_pred)))
    linear_r2 = float(r2_score(y_test, linear_test_pred))

    rf_mae = float(mean_absolute_error(y_test, rf_test_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_test_pred)))
    rf_r2 = float(r2_score(y_test, rf_test_pred))

    hybrid_mae = float(mean_absolute_error(y_test, hybrid_test_pred))
    hybrid_rmse = float(np.sqrt(mean_squared_error(y_test, hybrid_test_pred)))
    hybrid_r2 = float(r2_score(y_test, hybrid_test_pred))

    n = len(y_test)
    p = X.shape[1]

    linear_adj_r2 = adjusted_r2_score(linear_r2, n, p)
    hybrid_adj_r2 = adjusted_r2_score(hybrid_r2, n, p)

    metrics = {
        "Linear_MAE": linear_mae,
        "Linear_RMSE": linear_rmse,
        "Linear_R2": linear_r2,
        "Linear_Adjusted_R2": float(linear_adj_r2) if not np.isnan(linear_adj_r2) else None,

        "RF_MAE": rf_mae,
        "RF_RMSE": rf_rmse,
        "RF_R2": rf_r2,

        "Hybrid_MAE": hybrid_mae,
        "Hybrid_RMSE": hybrid_rmse,
        "Hybrid_R2": hybrid_r2,
        "Hybrid_Adjusted_R2": float(hybrid_adj_r2) if not np.isnan(hybrid_adj_r2) else None,
    }

    model_bundle = {
        "preprocessor": preprocessor,
        "linear_model": linear_model,
        "rf_model": rf_model,
        "linear_weight": linear_weight,
        "rf_weight": rf_weight,
    }

    return model_bundle, metrics, X.columns.tolist()


def predict_with_hybrid_breakup(model_bundle, new_df: pd.DataFrame, target_name: str):
    df_input = build_features(new_df.copy())
    X_processed = model_bundle["preprocessor"].transform(df_input)

    linear_pred = model_bundle["linear_model"].predict(X_processed)
    rf_pred = model_bundle["rf_model"].predict(X_processed)

    hybrid_pred = (
        model_bundle["linear_weight"] * linear_pred
        + model_bundle["rf_weight"] * rf_pred
    )

    result_df = new_df.copy()
    result_df[f"linear_{target_name}"] = np.round(linear_pred, 2)
    result_df[f"random_forest_{target_name}"] = np.round(rf_pred, 2)
    result_df[f"hybrid_predicted_{target_name}"] = np.round(hybrid_pred, 2)

    return result_df


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Upload training data")
training_file = st.sidebar.file_uploader("Upload CSV training file", type=["csv"])

linear_weight = st.sidebar.slider("Linear Regression Weight", 0.0, 1.0, 0.4, 0.05)
rf_weight = 1.0 - linear_weight
st.sidebar.write(f"Random Forest Weight: {rf_weight:.2f}")

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

if st.button("Train hybrid model"):
    try:
        model_bundle, metrics, feature_columns = train_hybrid_model(
            df, target_col, linear_weight=linear_weight, rf_weight=rf_weight
        )

        st.session_state["model_bundle"] = model_bundle
        st.session_state["feature_columns"] = feature_columns
        st.session_state["target_col"] = target_col

        st.success("Hybrid model trained successfully.")

        st.subheader("Linear Regression Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{metrics['Linear_MAE']:.3f}")
        c2.metric("RMSE", f"{metrics['Linear_RMSE']:.3f}")
        c3.metric("R²", f"{metrics['Linear_R2']:.3f}")
        c4.metric(
            "Adjusted R²",
            f"{metrics['Linear_Adjusted_R2']:.3f}" if metrics["Linear_Adjusted_R2"] is not None else "N/A"
        )

        st.subheader("Random Forest Performance")
        c5, c6, c7 = st.columns(3)
        c5.metric("MAE", f"{metrics['RF_MAE']:.3f}")
        c6.metric("RMSE", f"{metrics['RF_RMSE']:.3f}")
        c7.metric("R²", f"{metrics['RF_R2']:.3f}")

        st.subheader("Final Hybrid Performance")
        c8, c9, c10, c11 = st.columns(4)
        c8.metric("MAE", f"{metrics['Hybrid_MAE']:.3f}")
        c9.metric("RMSE", f"{metrics['Hybrid_RMSE']:.3f}")
        c10.metric("R²", f"{metrics['Hybrid_R2']:.3f}")
        c11.metric(
            "Adjusted R²",
            f"{metrics['Hybrid_Adjusted_R2']:.3f}" if metrics["Hybrid_Adjusted_R2"] is not None else "N/A"
        )

        st.info(
            f"Final hybrid prediction = {linear_weight:.2f} × Linear Regression + {rf_weight:.2f} × Random Forest"
        )

        model_bytes = io.BytesIO()
        joblib.dump(model_bundle, model_bytes)
        st.download_button(
            label="Download trained hybrid model",
            data=model_bytes.getvalue(),
            file_name="sugarcane_hybrid_yield_model.joblib",
            mime="application/octet-stream",
        )

    except Exception as e:
        st.error(f"Training failed: {e}")

if "model_bundle" in st.session_state:
    st.subheader("Predict on new data")
    st.write(
        "Upload a CSV with the same predictor columns as the training data, excluding the target column."
    )

    predict_file = st.file_uploader(
        "Upload CSV for prediction", type=["csv"], key="predict"
    )

    if predict_file is not None:
        try:
            new_df = pd.read_csv(predict_file)
            st.write("Prediction input preview")
            st.dataframe(new_df.head(), use_container_width=True)

            result_df = predict_with_hybrid_breakup(
                st.session_state["model_bundle"],
                new_df,
                st.session_state["target_col"],
            )

            st.subheader("Prediction results with model breakup")
            st.dataframe(result_df, use_container_width=True)

            pred_col = f"hybrid_predicted_{st.session_state['target_col']}"

            st.subheader("Prediction component summary")
            st.dataframe(
                result_df[
                    [
                        f"linear_{st.session_state['target_col']}",
                        f"random_forest_{st.session_state['target_col']}",
                        pred_col,
                    ]
                ],
                use_container_width=True,
            )

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_out,
                file_name="sugarcane_yield_predictions_with_model_breakup.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Train the hybrid model first to enable prediction.")