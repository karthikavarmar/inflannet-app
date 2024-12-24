import streamlit as st
from keras.models import load_model
import sys, os
import importlib.util
import pickle

def load_ml_model(module_path):
    # Load the ml_model module from the specified path
    spec = importlib.util.spec_from_file_location("ml_model", module_path)
    ml_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_model)
    return ml_model

def main():
    st.title("Influenza Epitope Prediction Tool")
    st.text("Please cite this paper InflANNet: https://doi.org/10.1186/s42269-023-01101-1")
    st.image("virus-image.jpg", width=150)

    # Load the trained model
    lmodel_9mer = load_model('tool_model_keras_9mer.h5')
    lmodel_15mer = load_model('tool_model_keras_15mer.h5')  # Replace with your model filename

    # Load the ml_model code as it's a Python file
    ml_model_path = "ml_functions.py"  # Replace with the actual path
    ml_model = load_ml_model(ml_model_path)

    # User input for the HTL(9-mer) epitope string
    input_string = st.text_input("Enter a 9 letter epitope string:")
    if st.button("Predict CTL score"):
        if input_string:
            prediction = ml_model.predict_epitope(lmodel_9mer, input_string)
            st.write(f"Prediction Score: {prediction}")
        else:
            st.write("Please enter an epitope string.")

    # User input for the CTL(6-mer) epitope string
    input_string = st.text_input("Enter a 15 letter epitope string:")   
    if st.button("Predict HTL score"):
        if input_string:
            prediction = ml_model.predict_epitope(lmodel_15mer, input_string)
            st.write(f"Prediction Score: {prediction}")
        else:
            st.write("Please enter an epitope string.")

if __name__ == '__main__':
    main()

