import os

def setup_mlflow():
    os.system("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5001")

if __name__ == "__main__":
    setup_mlflow()