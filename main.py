import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("energy_consumption.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Extract time features
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day_name()

print("Dataset Preview:")
print(df.head())

# ================================
# 1. Time Series Consumption Plot
# ================================
plt.figure()
plt.plot(df["Datetime"], df["Consumption_kWh"])
plt.xlabel("Time")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Electricity Consumption Over Time")
plt.show()

# ================================
# 2. Peak vs Off-Peak Analysis
# ================================
peak_hours = df[(df["Hour"] >= 18) & (df["Hour"] <= 22)]
off_peak_hours = df[(df["Hour"] < 6)]

print("\nAverage Peak Consumption:", peak_hours["Consumption_kWh"].mean())
print("Average Off-Peak Consumption:", off_peak_hours["Consumption_kWh"].mean())

# ================================
# 3. Rolling Average Trend
# ================================
df["Rolling_Avg"] = df["Consumption_kWh"].rolling(window=5).mean()

plt.figure()
plt.plot(df["Datetime"], df["Consumption_kWh"], label="Actual")
plt.plot(df["Datetime"], df["Rolling_Avg"], label="Rolling Avg")
plt.xlabel("Time")
plt.ylabel("Consumption (kWh)")
plt.title("Rolling Average Load Trend")
plt.legend()
plt.show()

# ================================
# 4. Weekday vs Weekend Comparison
# ================================
df["Type"] = np.where(df["Day"].isin(["Saturday", "Sunday"]), "Weekend", "Weekday")

weekday_avg = df[df["Type"] == "Weekday"]["Consumption_kWh"].mean()
weekend_avg = df[df["Type"] == "Weekend"]["Consumption_kWh"].mean()

plt.figure()
plt.bar(["Weekday", "Weekend"], [weekday_avg, weekend_avg])
plt.ylabel("Average Consumption (kWh)")
plt.title("Weekday vs Weekend Energy Usage")
plt.show()

# ================================
# 5. Anomaly Detection (Statistical)
# ================================
mean_load = df["Consumption_kWh"].mean()
std_load = df["Consumption_kWh"].std()

df["Anomaly"] = np.where(
    (df["Consumption_kWh"] > mean_load + 2 * std_load),
    "High Spike",
    "Normal"
)

print("\nDetected Anomalies:")
print(df[df["Anomaly"] == "High Spike"])

# ================================
# 6. Energy Efficiency Score
# ================================
max_load = df["Consumption_kWh"].max()
df["Efficiency_Score"] = 1 - (df["Consumption_kWh"] / max_load)

print("\nEnergy Efficiency Scores:")
print(df[["Datetime", "Efficiency_Score"]].head())
