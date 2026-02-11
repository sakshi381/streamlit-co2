import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="CO₂ Emissions Analysis",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------------
# SAMPLE DATA
# ----------------------------------
sample_data = pd.DataFrame({
    "Engine Size(L)": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    "CO2 Emissions(g/km)": [145, 185, 210, 250, 275, 320, 350, 400, 450]
})

# ----------------------------------
# TITLE & DESCRIPTION
# ----------------------------------
st.title("🚗 CO₂ Emissions vs Engine Size")
st.markdown("""
This application shows how **CO₂ emissions increase or decrease automatically**  
when **engine size (L)** changes using **Linear Regression**.
""")

# ----------------------------------
# SIDEBAR – DATA SOURCE
# ----------------------------------
st.sidebar.header("📂 Data Source")
option = st.sidebar.radio(
    "Select Data Input",
    ["Use Sample Data", "Upload CSV"]
)

# ----------------------------------
# LOAD DATA
# ----------------------------------
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        required_cols = {"Engine Size(L)", "CO2 Emissions(g/km)"}
        if not required_cols.issubset(data.columns):
            st.error("CSV must contain required columns")
            st.stop()

        data = data.dropna().drop_duplicates()
    else:
        st.warning("Upload a CSV file")
        st.stop()
else:
    data = sample_data

# ----------------------------------
# TRAIN REGRESSION MODEL
# ----------------------------------
X = data[["Engine Size(L)"]]
y = data["CO2 Emissions(g/km)"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------------
# SIDEBAR – ENGINE SIZE INPUT
# ----------------------------------
st.sidebar.header("⚙ Engine Control")
engine_size = st.sidebar.slider(
    "Engine Size (L)",
    float(data["Engine Size(L)"].min()),
    float(data["Engine Size(L)"].max()),
    step=0.1
)

# ----------------------------------
# PREDICTION
# ----------------------------------
predicted_co2 = model.predict([[engine_size]])[0]

st.subheader("📊 Predicted CO₂ Emissions")
st.metric("CO₂ (g/km)", f"{predicted_co2:.2f}")

# ----------------------------------
# REGRESSION LINE (AUTO-UPDATED)
# ----------------------------------
st.subheader("📈 Dynamic Regression Line")

engine_range = np.linspace(
    data["Engine Size(L)"].min(),
    data["Engine Size(L)"].max(),
    100
)

co2_range = model.predict(engine_range.reshape(-1, 1))

plt.figure(figsize=(8, 6))

# Scatter data
plt.scatter(
    data["Engine Size(L)"],
    data["CO2 Emissions(g/km)"],
    label="Dataset"
)

# Regression line
plt.plot(
    engine_range,
    co2_range,
    label="Regression Trend"
)

# Highlight selected engine size
plt.scatter(
    engine_size,
    predicted_co2,
    s=120,
    label="Selected Engine Size"
)

plt.xlabel("Engine Size (L)")
plt.ylabel("CO₂ Emissions (g/km)")
plt.title("CO₂ Emissions Automatically Change with Engine Size")
plt.legend()
plt.grid(True)

st.pyplot(plt)

# ----------------------------------
# BAR CHART (DATASET DEPENDENT)
# ----------------------------------
st.subheader("📊 CO₂ Emissions Comparison (Bar Chart)")

plt.figure(figsize=(8, 6))

plt.bar(
    data["Engine Size(L)"].astype(str),
    data["CO2 Emissions(g/km)"]
)

plt.xlabel("Engine Size (L)")
plt.ylabel("CO₂ Emissions (g/km)")
plt.title("CO₂ Emissions by Engine Size")
plt.grid(axis="y")

st.pyplot(plt)

# ----------------------------------
# INSIGHT
# ----------------------------------
st.subheader("🚀 Key Insight")
st.info(
    "Increasing engine size leads to higher fuel consumption, "
    "which automatically increases CO₂ emissions. "
    "Reducing engine size lowers emissions."
)

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("""
---
**Developed by Kalyani Sawkar**  
Powered by **Streamlit & Scikit-learn**
""")

