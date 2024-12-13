import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model import ActivityLevelClassifier
from data_loader import load_data
import mlflow
import mlflow.pytorch

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

        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('historical_beach_data.csv')
    train_model(X_train, y_train, X_test, y_test)