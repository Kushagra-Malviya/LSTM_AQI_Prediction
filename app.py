# 📚 Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 🎯 Streamlit page setup
st.set_page_config(page_title="AQI Forecast", layout="wide")

# 📦 Load trained model
model = load_model('Best.h5')  # Make sure Best.h5 is in the same folder as this app

# 📢 Title
st.title("🌍 Air Quality Index (AQI) 7-Day Forecast")
st.markdown("This app predicts AQI for the next 7 days based on recent pollution data.")

# 📈 Feature scaler (must match training time)
scaler = MinMaxScaler()

# ⚙️ Define pollutant features (as per your model)
features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'PRECTOTCORR']

# 🗂️ Input for last 7 days data
st.subheader("📅 Enter Latest 7 Days Pollution Data")

uploaded_file = st.file_uploader("Upload a CSV with last 7 days of feature values (columns: pm25, pm10, o3, no2, so2, co, PRECTOTCORR)", type=['csv'])

# If file uploaded
if uploaded_file is not None:
    # Read input
    input_df = pd.read_csv(uploaded_file)

    # Validation
    if input_df.shape != (7, 7):
        st.error("Please upload exactly 7 rows and 7 feature columns.")
    else:
        # Scale input
        scaler.fit(input_df)  # Fit scaler (assuming input is realistic)
        scaled_input = scaler.transform(input_df)

        # Predict 7 days into the future
        predictions = []
        current_sequence = scaled_input.copy()

        for _ in range(7):
            X_input = np.expand_dims(current_sequence, axis=0)  # (1, 7, 7)
            pred_aqi = model.predict(X_input)[0][0]
            predictions.append(pred_aqi)

            # Prepare input for next day
            # Dummy feature values assumed constant here (could be improved by modeling future pollution levels)
            next_features = current_sequence[-1, :]  # Take last day's features
            current_sequence = np.vstack([current_sequence[1:], next_features])  # Slide window

        # 📊 Show predictions
        st.subheader("🔮 AQI Predictions for Next 7 Days")

        # Convert predictions into a DataFrame
        days = pd.date_range(start=pd.Timestamp.today(), periods=7)
        aqi_df = pd.DataFrame({'Date': days.strftime("%d-%m-%Y"), 'Predicted AQI': predictions})

        st.dataframe(aqi_df.style.format({"Predicted AQI": "{:.2f}"}))

        # 📈 Plotting each day's AQI on a severity bar
        st.subheader("🚦 AQI Severity Visualization")

        for idx, row in aqi_df.iterrows():
            st.markdown(f"**{row['Date']}**")

            # AQI Severity bar
            fig, ax = plt.subplots(figsize=(8, 0.6))
            norm_value = row['Predicted AQI'] / 500  # normalize 0-1
            ax.barh(0, norm_value, color=plt.cm.RdYlGn_r(norm_value))
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(f"AQI: {row['Predicted AQI']:.2f}", fontsize=12)
            ax.axvline(0.5, color='gray', linestyle='--', lw=1)  # Optional midpoint line
            st.pyplot(fig)

else:
    st.info("👆 Please upload your latest 7 days pollution data as CSV to begin prediction.")
