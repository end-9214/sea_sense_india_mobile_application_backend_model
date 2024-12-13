# Beach Activity Level Prediction

This package uses ZenML and MLflow to train and track a model that predicts the activity level at a beach based on various environmental factors.

## Setup

1. Install the required packages:
    ```bash
    pip install zenml mlflow torch pandas scikit-learn
    ```

2. Run the setup script to initialize ZenML and start the MLflow server:
    ```bash
    python setup.py
    ```

## Running the Pipeline

To run the pipeline, execute the following command:
```bash
python pipeline.py
```
This will load the data, train the model, and log the metrics and model artifacts to MLflow.

### Instructions for Use

1. Clone the repository.
2. Install the required packages.
3. Run `setup.py` to set up ZenML and MLflow.
4. Run `pipeline.py` to execute the pipeline.