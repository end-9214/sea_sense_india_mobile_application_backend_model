import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date & Time'] = pd.to_datetime(data['Date & Time'])
    data['Hour'] = data['Date & Time'].dt.hour
    data['dayOfweek'] = data['Date & Time'].dt.dayofweek

    features = ['Sea Surface Temp (°C)', 'Air Temp (°C)', 'Wind Speed (km/h)', 'Wave Height (m)', 'UV Index', 'Hour', 'dayOfweek']
    X = data[features]
    y = data['Activity Level']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, './machine_learning_model/scaler.joblib')
    joblib.dump(label_encoder, './machine_learning_model/label_encoder.joblib')

    return X_train_scaled, X_test_scaled, y_train, y_test