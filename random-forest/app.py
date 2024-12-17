import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set the page title
st.title("Random Forest Regression: Height to Weight Prediction")

# Debugging: Print the path to check if the model file existstt
model_path = './static/random_forest_model.pkl'
if os.path.exists(model_path):
    st.write(f"Model file found at: {os.path.abspath(model_path)}")
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found at: {os.path.abspath(model_path)}")

# Sidebar for user input
st.sidebar.header("User Input")
height = st.sidebar.number_input("Enter Height (in inches):", min_value=50.0, max_value=90.0, value=65.0)

# Predict weight based on height
if st.sidebar.button("Predict"):
    prediction = model.predict([[height]])
    st.write(f"### Predicted Weight: {prediction[0]:.2f} lbs")

# Dataset display (optional)
if st.checkbox("Show Dataset"):
    dataset_url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
    data = pd.read_csv(dataset_url)
    st.write("### Sample Dataset")
    st.write(data.head())

# Visualization
if st.checkbox("Show Actual vs Predicted Plot"):
    dataset_url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
    data = pd.read_csv(dataset_url)
    data.columns = ['Index', 'Height', 'Weight']
    X = data[['Height']]
    y = data['Weight']
    y_pred = model.predict(X)

    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual", color='blue')
    ax.scatter(X, y_pred, label="Predicted", color='green', alpha=0.6)
    ax.set_title("Actual vs Predicted Values")
    ax.set_xlabel("Height (in inches)")
    ax.set_ylabel("Weight (in lbs)")
    ax.legend()
    st.pyplot(fig)
