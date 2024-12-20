import mlflow
import mlflow.pytorch
from data_loader import load_data
from train import train_model

def data_loading_step():
    return load_data('historical_beach_data.csv')

def training_step(data):
    X_train, X_test, y_train, y_test = data
    train_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    # Set or create an MLflow experiment
    mlflow.set_experiment("beach_activity_experiment")
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        data = data_loading_step()
        training_step(data)