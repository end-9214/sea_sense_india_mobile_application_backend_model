import torch
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



def predict_activity_level(new_sample):
    """
    Predict the activity level for a new sample.
    
    Parameters:
    - new_sample: A dictionary containing the sample data.
    - features: A list of feature names.
    - scaler: The scaler used to normalize the data.
    - model: The trained PyTorch model.
    - label_encoder: The label encoder used to transform labels.
    
    Returns:
    - predicted_label: The predicted activity level.

    """
    features = ['Sea Surface Temp (°C)', 'Air Temp (°C)', 'Wind Speed (km/h)', 'Wave Height (m)', 'UV Index', 'Hour', 'dayOfweek']
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    model = torch.load("./machine_learning_model./complete_beach_activity_model.pth")
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