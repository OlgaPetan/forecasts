import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
from pandas.tseries.offsets import MonthBegin
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import seasonal_decompose

# ----------------------------
# Load default or user data
# ----------------------------
@st.cache_data
def load_default_data():
    df = pd.read_excel('data.xlsx')
    df = df.fillna(0)
    df = df.melt(id_vars=["Product"], var_name="Date", value_name="Price")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.pivot(index='Date', columns='Product', values='Price').reset_index()
    df = df[df['Germany Road Freight Transport'] != 0]
    return df

st.title("📊 Forecast Germany Road Freight Transport")

uploaded_file = st.sidebar.file_uploader("Upload your own CSV file (optional):", type=["csv"])
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    try:
        user_df["Date"] = pd.to_datetime(user_df["Date"])
        df = user_df.set_index("Date")
        st.success("✅ Custom CSV loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error reading your file: {e}")
        st.stop()
else:
    df = load_default_data()
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

# State reset workaround
if 'prev_model' not in st.session_state:
    st.session_state.prev_model = None

# User parameters
all_features = [col for col in df.columns if col != "Germany Road Freight Transport"]
target_col = "Germany Road Freight Transport"
forecast_horizon = 6

st.sidebar.header("🛠️ Forecast Setup")
model_choice = st.sidebar.selectbox("Choose forecasting algorithm:", ["Business Analytiq AI", "Ridge Regression"])

if st.session_state.prev_model and st.session_state.prev_model != model_choice:
    st.experimental_rerun()

st.session_state.prev_model = model_choice

selected_features = st.sidebar.multiselect("Select independent variables (max 8):", all_features)

if len(selected_features) > 8:
    st.sidebar.error("You can select up to 8 independent variables.")
    st.stop()

lookback_months = st.sidebar.slider("How many months of history to consider?", 1, 6, 2)

if not selected_features:
    st.warning("Please select at least one independent variable to continue.")
    st.stop()

run_forecast = st.sidebar.button("🚀 Run Forecast")

if run_forecast:

    # Feature Engineering
    def create_lags(data, features, max_lag):
        df_lagged = pd.DataFrame(index=data.index)
        for lag in range(1, max_lag + 1):
            for col in features:
                df_lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
        return df_lagged

    X = create_lags(df, selected_features, lookback_months)
    y = pd.DataFrame(index=df.index)
    for i in range(1, forecast_horizon + 1):
        y[f"{target_col}_t+{i}"] = df[target_col].shift(-i)

    df_model = pd.concat([X, y], axis=1).dropna()
    df_model.index = pd.to_datetime(df_model.index)

    X_model = df_model[X.columns].copy()
    Y_model = df_model[y.columns].copy()

    # Model
    if model_choice == "XGBoost":
        base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
    else:
        base_model = Ridge(alpha=1.0)

    model = MultiOutputRegressor(base_model)
    model.fit(X_model, Y_model)
    Y_pred = model.predict(X_model)

    # Forecast with chaining for curve shape
    X_forecast = X_model.iloc[[-1]].copy()
    future_preds = []
    for step in range(1, forecast_horizon + 1):
        next_pred = model.predict(X_forecast)[0][step - 1]
        future_preds.append(next_pred)
        for feat in selected_features:
            for lag in range(lookback_months, 1, -1):
                X_forecast[f"{feat}_lag{lag}"] = X_forecast[f"{feat}_lag{lag - 1}"]
            X_forecast[f"{feat}_lag1"] = df[feat].iloc[-1]  # simple assumption

    last_date = df_model.index[-1]
    forecast_dates = [last_date + MonthBegin(i) for i in range(1, forecast_horizon + 1)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_preds})

    # Plot
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

    # Forecast Table
    st.markdown("### 📅 Forecast Table")
    st.dataframe(forecast_df)

    # Feature Importance
    st.markdown("### ⭐ Feature Importance")
    importances = pd.DataFrame(index=X_model.columns)
    for i, est in enumerate(model.estimators_):
        if model_choice == "XGBoost":
            score = est.feature_importances_
        else:
            score = np.abs(est.coef_)
        importances[f"t+{i+1}"] = score

    importances["Mean Importance"] = importances.mean(axis=1)

    def rename_lags(name):
        if "_lag" in name:
            base, lag = name.rsplit("_lag", 1)
            return f"{base} (lag {lag} month{'s' if lag != '1' else ''})"
        return name

    importances.index = [rename_lags(idx) for idx in importances.index]
    importances = importances.sort_values("Mean Importance", ascending=False)
    st.dataframe(importances[["Mean Importance"]].style.format("{:.4f}"))

    # Chart
    st.markdown("### 📊 Feature Importance Chart")
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    importances["Mean Importance"].plot(kind="barh", ax=ax_imp, color="teal")
    ax_imp.set_title("Feature Importance")
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    # Accuracy Metrics
    mae = mean_absolute_error(Y_model, Y_pred)
    r2 = r2_score(Y_model, Y_pred)
    accuracy_pct = max(0.0, min(1.0, r2)) * 100

    st.markdown("### 📈 Accuracy Metrics")
    st.write(f"**Estimated Forecast Accuracy:** {accuracy_pct:.1f}%")

    # Seasonality and Trend
    st.markdown("### 📉 Seasonality and Trend Decomposition")
    try:
        decomposition = seasonal_decompose(df[target_col], model='additive', period=12)
        fig_trend, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(decomposition.trend)
        axs[0].set_title("Trend")
        axs[1].plot(decomposition.seasonal)
        axs[1].set_title("Seasonality")
        axs[2].plot(decomposition.resid)
        axs[2].set_title("Residuals")
        plt.tight_layout()
        st.pyplot(fig_trend)
    except Exception as e:
        st.warning(f"Could not decompose seasonality/trend: {e}")
