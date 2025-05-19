import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthBegin

st.set_page_config(layout="wide")
st.title("📈 6-Month Forecasting App (Linear Regression)")

# ----------------------------
# Load local Excel file
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel('data.xlsx')
    df = df.fillna(0)
    df = df.melt(id_vars=["Product"], var_name="Date", value_name="Price")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.pivot(index='Date', columns='Product', values='Price').reset_index()
    df = df[df['Germany Road Freight Transport'] != 0]
    return df

df = load_data()
df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

target_col = "Germany Road Freight Transport"
all_features = [col for col in df.columns if col != target_col]
forecast_horizon = 6

# ----------------------------
# Feature selection
# ----------------------------
st.sidebar.header("🔧 Model Setup")
selected_features = st.sidebar.multiselect("Select features for the model:", all_features)

if not selected_features:
    st.warning("Please select at least one feature from the sidebar.")
    st.stop()

run_forecast = st.button("🚀 Run Forecast")

if run_forecast:
    # ----------------------------
    # Create lag features
    # ----------------------------
    def create_lags(data, features, lags=[1, 2]):
        df_lagged = pd.DataFrame(index=data.index)
        for lag in lags:
            for col in features:
                df_lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
        return df_lagged

    X = create_lags(df, selected_features)
    y = pd.DataFrame(index=df.index)
    for i in range(1, forecast_horizon + 1):
        y[f"{target_col}_t+{i}"] = df[target_col].shift(-i)

    df_model = pd.concat([X, y], axis=1).dropna()
    df_model.index = pd.to_datetime(df_model.index)

    X_model = df_model[X.columns].copy()
    if isinstance(X_model, pd.Series):
        X_model = X_model.to_frame()

    Y_model = df_model[y.columns].copy()

    # ----------------------------
    # Train model
    # ----------------------------
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_model, Y_model)
    Y_pred = model.predict(X_model)

    # ----------------------------
    # Forecast
    # ----------------------------
    X_latest = X_model.iloc[[-1]]
    if isinstance(X_latest, pd.Series):
        X_latest = X_latest.to_frame().T

    future_preds = model.predict(X_latest)[0]
    last_date = df_model.index[-1]
    forecast_dates = [last_date + MonthBegin(i) for i in range(1, forecast_horizon + 1)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_preds})

    # ----------------------------
    # Plot FIRST
    # ----------------------------
    st.markdown("### 📈 Forecast vs Historical")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_model.index, df.loc[df_model.index, target_col], label="Historical")
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], marker='o', label="Forecast")
    ax.axvline(df_model.index[-1], linestyle='--', color='gray', alpha=0.6, label="Forecast Start")
    ax.set_title("Germany Road Freight Transport Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ----------------------------
    # Forecast Table
    # ----------------------------
    st.markdown("### 📅 Forecast Table")
    st.dataframe(forecast_df)

    # ----------------------------
    # Feature Importance
    # ----------------------------
    st.markdown("### ⭐ Feature Importance")

    importances = pd.DataFrame(index=X_model.columns)
    for i, est in enumerate(model.estimators_):
        importances[f"t+{i+1}"] = np.abs(est.coef_)
    importances["Mean Importance"] = importances.mean(axis=1)
    importances = importances.sort_values("Mean Importance", ascending=False)

    st.dataframe(importances[["Mean Importance"]].style.format("{:.4f}"))

    # Bar chart
    st.markdown("### 📊 Feature Importance Chart")
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    importances["Mean Importance"].plot(kind="barh", ax=ax_imp, color="teal")
    ax_imp.set_xlabel("Mean Absolute Coefficient")
    ax_imp.set_ylabel("Feature (lagged)")
    ax_imp.set_title("Feature Importance (Mean Across Forecast Horizons)")
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    # ----------------------------
    # Accuracy Metrics
    # ----------------------------
    mae = mean_absolute_error(Y_model, Y_pred)
    r2 = r2_score(Y_model, Y_pred)

    st.markdown("### 📈 Accuracy Metrics")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    # ----------------------------
    # Bootstrapped Coefficients
    # ----------------------------
    @st.cache_data
    def bootstrap_confidence_intervals(X, Y, _model_cls, n_bootstraps=500, alpha=0.05):
        coefs = []
        for _ in range(n_bootstraps):
            X_resampled, Y_resampled = resample(X, Y)
            model = model_cls()
            model.fit(X_resampled, Y_resampled)
            coefs.append([est.coef_ for est in model.estimators_])

        coefs = np.array(coefs)
        lower = np.percentile(coefs, 100 * alpha / 2, axis=0)
        upper = np.percentile(coefs, 100 * (1 - alpha / 2), axis=0)
        mean = np.mean(coefs, axis=0)
        return mean, lower, upper

    st.markdown("### 🧮 Coefficients + Confidence Intervals")

    mean_coef, lower, upper = bootstrap_confidence_intervals(
        X_model, Y_model, lambda: MultiOutputRegressor(LinearRegression())
    )

    coef_df = pd.DataFrame(index=X_model.columns)
    for i in range(forecast_horizon):
        coef_df[f"t+{i+1} Mean"] = mean_coef[i]
        coef_df[f"t+{i+1} Lower"] = lower[i]
        coef_df[f"t+{i+1} Upper"] = upper[i]

    st.dataframe(coef_df.style.format("{:.3f}"))
