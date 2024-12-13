import os

def setup_zenml():
    os.system("zenml init")
    os.system("zenml integration install pytorch -y")
    os.system("zenml integration install mlflow -y")
    os.system("zenml experiment-tracker register mlflow_tracker --type=mlflow --config=uri=http://127.0.0.1:5000")
    os.system("zenml experiment-tracker set mlflow_tracker")

def setup_mlflow():
    os.system("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")

if __name__ == "__main__":
    setup_zenml()
    setup_mlflow()