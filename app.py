import streamlit as st
from keras.models import load_model
import sys, os
import importlib.util
import pickle
#import pandas as py

def main():
    st.title("Flu Epitope Prediction Tool")

    # Load the trained model
    lmodel = load_model('tool_model_keras_9mer.h5')

    # Load the ml_model code if it's a Python file
    ml_model_path = "ml_functions.py"  # Replace with the actual path
    ml_model = load_ml_model(ml_model_path)

    # User input for the epitope string
    input_string = st.text_input("Enter a 9 letter epitope string:")

    if st.button("Predict"):
        if input_string:
            prediction = ml_model.predict_epitope(lmodel, input_string)
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Please enter an epitope string.")

if __name__ == '__main__':
    main()
