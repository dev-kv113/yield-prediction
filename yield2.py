import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Sugarcane Yield Prediction", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.image("logo copy.png", width=200)
st.markdown(
    """
    <h2 style="margin-top: -19px; margin-bottom: -6px;">
        AI Based Advanced Sugarcane Yield Prediction Model
    </h2>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<p style="margin-bottom: 0.5px;">
Upload historical data, compare Linear Regression, Random Forest, and Hybrid model performance, then predict yield for fresh input data.
</p>
<p style="margin-top: 0px;">
Proprietary AI based yield prediction model developed by Geotrans Technologies Pvt. Ltd.
</p>
""", unsafe_allow_html=True)

# ---------------------------------
# Helper functions
# ---------------------------------
def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, p):
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    adj_r2_val = adjusted_r2(test_r2, len(y_test), p)

    return {
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Adjusted R²": adj_r2_val,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Gap": train_r2 - test_r2
    }


def interpret_model(train_r2, test_r2):
    gap = train_r2 - test_r2
    if test_r2 < 0:
        return "Poor: Test R² is negative. Model performs worse than predicting the mean."
    elif gap > 0.15:
        return f"Possible overfitting: Train-Test gap is high ({gap:.4f})."
    else:
        return "Reasonably stable: Train and Test performance are fairly close."


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def prepare_encoded_training_data(df, feature_columns, target_column):
    model_df = df[feature_columns + [target_column]].copy()

    # convert target to numeric safely
    model_df[target_column] = pd.to_numeric(model_df[target_column], errors="coerce")

    # drop rows where target is missing
    model_df = model_df.dropna(subset=[target_column])

    # split raw X and y
    X_raw = model_df[feature_columns].copy()
    y = model_df[target_column].copy()

    # drop rows with missing feature values
    valid_idx = X_raw.dropna().index
    X_raw = X_raw.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    if X_raw.shape[0] < 10:
        raise ValueError("Not enough valid rows after removing missing values.")

    # detect categorical columns
    categorical_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # one-hot encode
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    # force all columns numeric
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.astype(float)

    # final safety cleanup
    valid_idx_final = X_encoded.dropna().index
    X_encoded = X_encoded.loc[valid_idx_final].copy()
    y = y.loc[valid_idx_final].copy()

    if X_encoded.shape[0] < 10:
        raise ValueError("Not enough valid rows after encoding and numeric conversion.")

    non_numeric_after_encoding = X_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_after_encoding:
        raise ValueError(f"These columns are still non-numeric after encoding: {non_numeric_after_encoding}")

    return X_encoded, y, categorical_cols


def prepare_encoded_prediction_data(pred_df, raw_feature_columns, trained_feature_columns, categorical_cols):
    pred_input = pred_df[raw_feature_columns].copy()

    # one-hot encode using same categorical columns
    pred_input = pd.get_dummies(pred_input, columns=categorical_cols, drop_first=True)

    # add missing columns seen during training
    for col in trained_feature_columns:
        if col not in pred_input.columns:
            pred_input[col] = 0.0

    # remove extra unseen columns
    extra_cols = [col for col in pred_input.columns if col not in trained_feature_columns]
    if extra_cols:
        pred_input = pred_input.drop(columns=extra_cols)

    # reorder
    pred_input = pred_input[trained_feature_columns]

    # force numeric
    pred_input = pred_input.apply(pd.to_numeric, errors="coerce")
    pred_input = pred_input.astype(float)

    non_numeric_pred_cols = pred_input.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_pred_cols:
        raise ValueError(f"These prediction columns are still non-numeric: {non_numeric_pred_cols}")

    return pred_input


# ---------------------------------
# Upload training data
# ---------------------------------
st.header("1. Upload Training Data")
train_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"])

if train_file is not None:
    try:
        df = pd.read_csv(train_file)

        if df.empty:
            st.error("The uploaded file is empty.")
            st.stop()

        st.subheader("Training Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.header("2. Select Target and Feature Columns")
        target_column = st.selectbox("Select target column", df.columns)

        feature_options = [col for col in df.columns if col != target_column]
        feature_columns = st.multiselect(
            "Select feature columns",
            feature_options,
            default=feature_options
        )

        if not feature_columns:
            st.warning("Please select at least one feature column.")
            st.stop()

        st.header("3. Model Settings")
        c1, c2, c3 = st.columns(3)

        with c1:
            test_size = st.slider("Test size (%)", 10, 40, 20, 5)

        with c2:
            n_estimators = st.slider("Random Forest trees", 50, 500, 200, 50)

        with c3:
            max_depth_option = st.selectbox("Random Forest max depth", ["None", 5, 10, 15, 20])

        max_depth = None if max_depth_option == "None" else int(max_depth_option)

        st.header("4. Train Models")
        if st.button("Train and Compare Models"):

            # ---------------------------------
            # Prepare data
            # ---------------------------------
            X, y, categorical_cols = prepare_encoded_training_data(df, feature_columns, target_column)

            if categorical_cols:
                st.info(f"Categorical columns detected and encoded: {categorical_cols}")

            trained_feature_columns = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            p = X_train.shape[1]

            # -----------------------------
            # 1. Linear Regression
            # -----------------------------
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)

            y_train_pred_lin = lin_model.predict(X_train)
            y_test_pred_lin = lin_model.predict(X_test)

            lin_metrics = evaluate_model(y_train, y_train_pred_lin, y_test, y_test_pred_lin, p)

            # -----------------------------
            # 2. Random Forest
            # -----------------------------
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                oob_score=True,
                bootstrap=True
            )
            rf_model.fit(X_train, y_train)

            y_train_pred_rf = rf_model.predict(X_train)
            y_test_pred_rf = rf_model.predict(X_test)

            rf_metrics = evaluate_model(y_train, y_train_pred_rf, y_test, y_test_pred_rf, p)
            rf_metrics["OOB R²"] = rf_model.oob_score_

            # -----------------------------
            # 3. Hybrid model
            # Linear + RF on residuals
            # -----------------------------
            residual_train = y_train - y_train_pred_lin

            rf_residual_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            rf_residual_model.fit(X_train, residual_train)

            residual_train_pred = rf_residual_model.predict(X_train)
            residual_test_pred = rf_residual_model.predict(X_test)

            y_train_pred_hybrid = y_train_pred_lin + residual_train_pred
            y_test_pred_hybrid = y_test_pred_lin + residual_test_pred

            hybrid_metrics = evaluate_model(y_train, y_train_pred_hybrid, y_test, y_test_pred_hybrid, p)

            # -----------------------------
            # Comparison table
            # -----------------------------
            comparison_df = pd.DataFrame({
                "Model": ["Linear Regression", "Random Forest", "Hybrid"],
                "Train R²": [
                    lin_metrics["Train R²"],
                    rf_metrics["Train R²"],
                    hybrid_metrics["Train R²"]
                ],
                "Test R²": [
                    lin_metrics["Test R²"],
                    rf_metrics["Test R²"],
                    hybrid_metrics["Test R²"]
                ],
                "Adjusted R²": [
                    lin_metrics["Adjusted R²"],
                    rf_metrics["Adjusted R²"],
                    hybrid_metrics["Adjusted R²"]
                ],
                "Test MAE": [
                    lin_metrics["Test MAE"],
                    rf_metrics["Test MAE"],
                    hybrid_metrics["Test MAE"]
                ],
                "Test RMSE": [
                    lin_metrics["Test RMSE"],
                    rf_metrics["Test RMSE"],
                    hybrid_metrics["Test RMSE"]
                ],
                "Overfit Gap": [
                    lin_metrics["Gap"],
                    rf_metrics["Gap"],
                    hybrid_metrics["Gap"]
                ],
                "OOB R²": [
                    np.nan,
                    rf_metrics.get("OOB R²", np.nan),
                    np.nan
                ],
                "Interpretation": [
                    interpret_model(lin_metrics["Train R²"], lin_metrics["Test R²"]),
                    interpret_model(rf_metrics["Train R²"], rf_metrics["Test R²"]),
                    interpret_model(hybrid_metrics["Train R²"], hybrid_metrics["Test R²"])
                ]
            })

            # best model based on highest Test R²
            best_model_name = comparison_df.sort_values("Test R²", ascending=False).iloc[0]["Model"]

            if best_model_name == "Linear Regression":
                best_model = lin_model
            elif best_model_name == "Random Forest":
                best_model = rf_model
            else:
                best_model = {
                    "linear_model": lin_model,
                    "residual_model": rf_residual_model
                }

            # save in session state
            st.session_state["comparison_df"] = comparison_df
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_model"] = best_model
            st.session_state["raw_feature_columns"] = feature_columns
            st.session_state["trained_feature_columns"] = trained_feature_columns
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["target_column"] = target_column
            st.session_state["rf_feature_importance"] = pd.DataFrame({
                "Feature": trained_feature_columns,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            # -----------------------------
            # Display results
            # -----------------------------
            st.header("5. Model Comparison")
            st.dataframe(comparison_df.round(4), use_container_width=True)

            st.success(f"Best model based on Test R²: {best_model_name}")

            best_row = comparison_df.sort_values("Test R²", ascending=False).iloc[0]
            b1, b2, b3 = st.columns(3)

            with b1:
                st.metric("Best Model", best_model_name)
            with b2:
                st.metric("Best Test R²", f"{best_row['Test R²']:.4f}")
            with b3:
                st.metric("Best Test RMSE", f"{best_row['Test RMSE']:.4f}")

            st.subheader("Random Forest Feature Importance")
            st.dataframe(st.session_state["rf_feature_importance"], use_container_width=True)
            st.bar_chart(st.session_state["rf_feature_importance"].set_index("Feature"))

            st.subheader("Actual vs Predicted on Test Data")
            results_test_df = pd.DataFrame({
                "Actual": y_test.values,
                "Linear Prediction": y_test_pred_lin,
                "RF Prediction": y_test_pred_rf,
                "Hybrid Prediction": y_test_pred_hybrid
            })
            st.dataframe(results_test_df.head(25), use_container_width=True)

    except Exception as e:
        st.error(f"Error reading or processing file: {e}")


# ---------------------------------
# Prediction section
# ---------------------------------
if "best_model" in st.session_state:
    st.header("6. Upload Fresh Input Data for Prediction")
    pred_file = st.file_uploader("Upload fresh input dataset (CSV)", type=["csv"], key="pred_file")

    if pred_file is not None:
        try:
            pred_df = pd.read_csv(pred_file)

            if pred_df.empty:
                st.error("The prediction file is empty.")
                st.stop()

            st.subheader("Fresh Input Data Preview")
            st.dataframe(pred_df.head(), use_container_width=True)

            raw_feature_columns = st.session_state["raw_feature_columns"]
            trained_feature_columns = st.session_state["trained_feature_columns"]
            categorical_cols = st.session_state["categorical_cols"]

            missing_raw_cols = [col for col in raw_feature_columns if col not in pred_df.columns]
            if missing_raw_cols:
                st.error(f"Missing required columns in prediction file: {missing_raw_cols}")
            else:
                pred_input_raw = pred_df[raw_feature_columns].copy()

                # remove rows with missing values in required raw features
                if pred_input_raw.isnull().sum().sum() > 0:
                    st.warning("Rows with missing values in prediction data will be removed.")
                    valid_idx = pred_input_raw.dropna().index
                    pred_input_raw = pred_input_raw.loc[valid_idx].copy()
                    pred_df = pred_df.loc[valid_idx].copy()

                pred_encoded = prepare_encoded_prediction_data(
                    pred_df=pred_input_raw,
                    raw_feature_columns=raw_feature_columns,
                    trained_feature_columns=trained_feature_columns,
                    categorical_cols=categorical_cols
                )

                # drop rows that became invalid after numeric conversion
                valid_idx_final = pred_encoded.dropna().index
                pred_encoded = pred_encoded.loc[valid_idx_final].copy()
                pred_df = pred_df.loc[valid_idx_final].copy()

                if pred_encoded.empty:
                    st.error("No valid rows left in prediction data after processing.")
                    st.stop()

                best_model_name = st.session_state["best_model_name"]
                best_model = st.session_state["best_model"]

                if best_model_name == "Linear Regression":
                    predictions = best_model.predict(pred_encoded)

                elif best_model_name == "Random Forest":
                    predictions = best_model.predict(pred_encoded)

                else:
                    linear_preds = best_model["linear_model"].predict(pred_encoded)
                    residual_preds = best_model["residual_model"].predict(pred_encoded)
                    predictions = linear_preds + residual_preds

                result_df = pred_df.copy()
                result_df["Predicted_Yield"] = predictions

                st.subheader("Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                csv_data = to_csv(result_df)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="predicted_yield_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading prediction file: {e}")