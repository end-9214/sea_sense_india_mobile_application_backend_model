import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ActivityLevelClassifier
import joblib

def predict_activity_level(new_sample):
    """
    Predict the activity level for a new sample.
    
    Parameters:
    - new_sample: A dictionary containing the sample data.
    
    Returns:
    - predicted_label: The predicted activity level.
    """
    features = ['Sea Surface Temp (°C)', 'Air Temp (°C)', 'Wind Speed (km/h)', 'Wave Height (m)', 'UV Index', 'Hour', 'dayOfweek']

    # Load the scaler and label encoder used during training
    scaler = joblib.load("./machine_learning_model/scaler.joblib")
    label_encoder = joblib.load("./machine_learning_model/label_encoder.joblib")

    # Load the model
    model = torch.load("./machine_learning_model/complete_beach_activity_model.pth", map_location=torch.device('cpu'))
    model.eval()

    # Convert to DataFrame
    new_sample_df = pd.DataFrame([new_sample])

    # Select and scale features
    new_sample_features = new_sample_df[features]
    new_sample_scaled = scaler.transform(new_sample_features)

    # Convert to tensor
    new_sample_tensor = torch.tensor(new_sample_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(new_sample_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = label_encoder.inverse_transform(predicted_class.numpy())
    
    return predicted_label[0]

# Example usage:
# new_sample = {
#     'Date & Time': '2024-12-20 00:00:00',
#     'Sea Surface Temp (°C)': 27.0,
#     'Air Temp (°C)': 22.0,
#     'Wind Speed (km/h)': 4.2,
#     'Wave Height (m)': 0.49,
#     'UV Index': 0.0,
#     'Hour': 0,
#     'dayOfweek': 5
# }
# predicted_label = predict_activity_level(new_sample)
# print(f'Predicted Activity Level: {predicted_label}')