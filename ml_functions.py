import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Define the amino acid codes
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Function to perform one-hot encoding on a peptide sequence
def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    x = pd.DataFrame(np.zeros((len(seq), len(o)), dtype=int), columns=o)
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values.flatten()
    return e

# Define a function for making predictions
def predict_epitope(model, input_string):
    # Preprocess the input_string using one-hot encoding
    encoded_input = one_hot_encode(input_string)
    
    # Convert the encoded input to the required shape (if necessary)
    encoded_input = np.array(encoded_input).reshape(1, -1)
    
    # Use the loaded model to make predictions
    prediction = model.predict(encoded_input)
    
    # Return the prediction
    return prediction[0, 1]