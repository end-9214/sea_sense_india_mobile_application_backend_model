import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ActivityLevelClassifier
from data_loader import load_data
import mlflow
import mlflow.pytorch
import numpy as np

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

class BeachActivityDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = len(set(y_train))

    model = ActivityLevelClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_dataset = BeachActivityDataset(X_train, y_train)
    test_dataset = BeachActivityDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    mlflow.pytorch.autolog()

    # End any active run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.numpy())
                y_pred.extend(predicted.numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model parameters
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("output_dim", output_dim)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("num_epochs", num_epochs)

        # Ensure the input example is float32
        example_input = np.array(X_test[0:1], dtype=np.float32)
        mlflow.pytorch.log_model(model, "model", input_example=example_input, pip_requirements=["torch==2.4.1+cpu"])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('historical_beach_data.csv')
    train_model(X_train, y_train, X_test, y_test)