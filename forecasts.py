import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ----------------------------
# Load built-in dataset
# ----------------------------

st.title("📈 6-Month Forecasting App (Linear Regression)")

# ----------------------------
# Load and preprocess data
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


st.markdown("### Forecast target:")
st.code(target_col, language="python")

st.markdown("### Available features:")
st.write(all_features)

# ----------------------------
# Sidebar: Select columns
# ----------------------------
st.sidebar.title("🔧 Configuration")
selected_features = st.sidebar.multiselect("Select features to include in the forecast model:", all_features, default=all_features)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

# ----------------------------
# Create lagged feature matrix
# ----------------------------

def create_lags(data, features, lags=[1, 2]):
    df_lagged = pd.DataFrame(index=data.index)
    for lag in lags:
        for col in features:
            df_lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
    return df_lagged

lags = [1, 2]
X = create_lags(df, selected_features, lags)
y = pd.DataFrame(index=df.index)
for i in range(1, forecast_horizon + 1):
    y[f"{target_col}_t+{i}"] = df[target_col].shift(-i)

df_model = pd.concat([X, y], axis=1).dropna()
X_model = df_model[X.columns]
Y_model = df_model[y.columns]

# ----------------------------
# Train Linear Regression
# ----------------------------

model = MultiOutputRegressor(LinearRegression())
model.fit(X_model, Y_model)
Y_pred = model.predict(X_model)

mae = mean_absolute_error(Y_model, Y_pred)
r2 = r2_score(Y_model, Y_pred)

# ----------------------------
# Forecast future
# ----------------------------

X_latest = X_model.iloc[[-1]]
future_preds = model.predict(X_latest)[0]
forecast_dates = pd.date_range(start=df_model.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_preds})

# ----------------------------
# Bootstrapped Confidence Intervals
# ----------------------------

def bootstrap_confidence_intervals(X, Y, model_cls, n_bootstraps=1000, alpha=0.05):
    lower_bounds = []
    upper_bounds = []
    coefs = []

    for _ in range(n_bootstraps):
        X_resampled, Y_resampled = resample(X, Y)
        model = model_cls()
        model.fit(X_resampled, Y_resampled)
        coefs.append([est.coef_ for est in model.estimators_])

    coefs = np.array(coefs)
    lower = np.percentile(coefs, 100 * alpha / 2, axis=0)
    upper = np.percentile(coefs, 100 * (1 - alpha / 2), axis=0)
    mean_coef = np.mean(coefs, axis=0)
    return mean_coef, lower, upper

st.markdown("### 🔍 Accuracy Metrics")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R²:** {r2:.2f}")

# ----------------------------
# Display Forecast
# ----------------------------

st.markdown("### 📅 Forecast for Next 6 Months")
st.dataframe(forecast_df)

# ----------------------------
# Plot Forecast
# ----------------------------

st.markdown("### 📊 Forecast vs Historical")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_model.index, df_model[target_col], label="Historical")
ax.plot(forecast_df['Date'], forecast_df['Forecast'], label="Forecast", marker='o')
ax.axvline(df_model.index[-1], linestyle='--', color='gray', alpha=0.6, label="Forecast Start")
ax.set_title("Germany Road Freight Transport Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Cost")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# Coefficients + Confidence Intervals
# ----------------------------

st.markdown("### 🧮 Coefficients and Confidence Intervals (Bootstrapped)")
mean_coef, lower, upper = bootstrap_confidence_intervals(X_model, Y_model, lambda: MultiOutputRegressor(LinearRegression()))

coef_df = pd.DataFrame(index=X_model.columns)
for i in range(forecast_horizon):
    coef_df[f"t+{i+1} Mean"] = mean_coef[i]
    coef_df[f"t+{i+1} Lower"] = lower[i]
    coef_df[f"t+{i+1} Upper"] = upper[i]

st.dataframe(coef_df.style.format("{:.3f}"))
