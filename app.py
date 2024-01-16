import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
num_labels = 20  # Update this with the correct number of labels
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.load_state_dict(torch.load(r'C:\Users\Admin\Desktop\F.M-TextClassificationUsingHuggingFaceModelBasedOnNACE\model.pkl', map_location=torch.device('cpu')))
device = torch.device('cpu')  # Change this to CPU
model.to(device)

# Define a dictionary to map numeric labels to NACE sector names
label_to_sector = {
    0: "Accommodation And Food Service Activities",
    1: "Activities Of Households As Employers; Undifferentiated Goods - And Services - Producing Activities Of Households For Own Use",
    2: "Administrative And Support Service Activities",
    3: "Agriculture, Forestry And Fishing",
    4: "Arts, Entertainment And Recreation",
    5: "Construction",
    6: "Education",
    7: "Electricity, Gas, Steam And Air Conditioning Supply",
    8: "Financial And Insurance Activities",
    9: "Human Health And Social Work Activities",
    10: "Information And Communication",
    11: "Manufacturing",
    12: "Mining And Quarrying",
    13: "Other Services Activities",
    14: "Professional, Scientific And Technical Activities",
    15: "Public Administration And Defence; Compulsory Social Security",
    16: "Real Estate Activities",
    17: "Transporting And Storage",
    18: "Water Supply; Sewerage; Waste Managment And Remediation Activities",
    19: "Wholesale And Retail Trade; Repair Of Motor Vehicles And Motorcycles"
}

# Function to make predictions
def predict_job(title_or_description):
    input_text = title_or_description
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    preds = torch.argmax(logits, axis=1)
    return preds.item()

# Streamlit UI
st.title("Job Text Classification")

title_or_description = st.text_area("Enter the job title or description:")
if st.button("Predict"):
    if title_or_description:
        prediction = predict_job(title_or_description)
        predicted_sector = label_to_sector.get(prediction, f"Unknown Sector {prediction}")
        st.success(f"Predicted NACE Sector: {predicted_sector}")
    else:
        st.warning("Please enter either the job title or description for prediction.")