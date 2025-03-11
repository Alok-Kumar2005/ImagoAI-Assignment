import streamlit as st
import requests

## setting the title of the web app
st.title("Machine Learning Prediction App")
st.write("Enter 20 feature values (each between 0 and 1):")

# getting the 20 input values from the user
# and storing them in a list
features = []
for i in range(20):
    value = st.number_input(
        label=f"Feature {i+1}",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=0.5,
        key=f"feature_{i}"
    )
    features.append(value)

if st.button("Predict"):
    payload = {"features": features}
    # URL of your FastAPI endpoint   ( change it accordingly )
    url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            prediction = response.json().get("prediction", None)
            st.success(f"Predicted Value: {prediction}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
