import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------
# App Title & Description
# --------------------------------
st.set_page_config(page_title="Energy Consumption Analyzer", layout="wide")

st.title("⚡ Energy Consumption & Load Analyzer")
st.write(
    "This application analyzes electricity consumption patterns to identify "
    "peak loads, trends, anomalies, and energy efficiency using data analytics."
)

# --------------------------------
# Upload CSV
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Energy Consumption CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Extract time features
    df["Hour"] = df["Datetime"].dt.hour
    df["Day"] = df["Datetime"].dt.day_name()

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------
    # Time Series Plot
    # --------------------------------
    st.subheader("📈 Electricity Consumption Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Datetime"], df["Consumption_kWh"])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Consumption (kWh)")
    ax1.set_title("Energy Consumption Time Series")
    st.pyplot(fig1)

    # --------------------------------
    # Peak vs Off-Peak Analysis
    # --------------------------------
    st.subheader("⏰ Peak vs Off-Peak Analysis")

    peak_hours = df[(df["Hour"] >= 18) & (df["Hour"] <= 22)]
    off_peak_hours = df[df["Hour"] < 6]

    st.write("**Average Peak Consumption (6 PM – 10 PM):**",
             round(peak_hours["Consumption_kWh"].mean(), 2))
    st.write("**Average Off-Peak Consumption (Before 6 AM):**",
             round(off_peak_hours["Consumption_kWh"].mean(), 2))

    # --------------------------------
    # Rolling Average Trend
    # --------------------------------
    st.subheader("📉 Rolling Average Load Trend")

    window = st.slider("Select Rolling Window Size", 2, 24, 5)
    df["Rolling_Avg"] = df["Consumption_kWh"].rolling(window=window).mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Datetime"], df["Consumption_kWh"], label="Actual Load")
    ax2.plot(df["Datetime"], df["Rolling_Avg"], label="Rolling Average")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Consumption (kWh)")
    ax2.set_title("Rolling Average Energy Load")
    ax2.legend()
    st.pyplot(fig2)

    # --------------------------------
    # Weekday vs Weekend Comparison
    # --------------------------------
    st.subheader("📊 Weekday vs Weekend Consumption")

    df["Type"] = np.where(
        df["Day"].isin(["Saturday", "Sunday"]),
        "Weekend",
        "Weekday"
    )

    weekday_avg = df[df["Type"] == "Weekday"]["Consumption_kWh"].mean()
    weekend_avg = df[df["Type"] == "Weekend"]["Consumption_kWh"].mean()

    fig3, ax3 = plt.subplots()
    ax3.bar(["Weekday", "Weekend"], [weekday_avg, weekend_avg])
    ax3.set_ylabel("Average Consumption (kWh)")
    ax3.set_title("Energy Usage Comparison")
    st.pyplot(fig3)

    # --------------------------------
    # Anomaly Detection
    # --------------------------------
    st.subheader("🚨 Anomaly Detection")

    mean_load = df["Consumption_kWh"].mean()
    std_load = df["Consumption_kWh"].std()

    threshold = st.slider("Select Anomaly Threshold (Std Dev)", 1.0, 3.0, 2.0)
    df["Anomaly"] = np.where(
        df["Consumption_kWh"] > mean_load + threshold * std_load,
        "High Spike",
        "Normal"
    )

    st.write("Detected High Consumption Spikes:")
    st.dataframe(df[df["Anomaly"] == "High Spike"])

    # --------------------------------
    # Energy Efficiency Score
    # --------------------------------
    st.subheader("🌱 Energy Efficiency Score")

    max_load = df["Consumption_kWh"].max()
    df["Efficiency_Score"] = 1 - (df["Consumption_kWh"] / max_load)

    st.write("Efficiency Score (Closer to 1 = More Efficient):")
    st.dataframe(df[["Datetime", "Efficiency_Score"]].head())

else:
    st.info("Please upload a CSV file to start analysis.")
