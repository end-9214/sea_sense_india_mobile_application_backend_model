from zenml.pipelines import pipeline
from zenml.steps import step
from data_loader import load_data
from train import train_model

@step
def data_loading_step():
    return load_data('historical_beach_data.csv')

@step
def training_step(data):
    X_train, X_test, y_train, y_test = data
    train_model(X_train, y_train, X_test, y_test)

@pipeline
def beach_activity_pipeline(data_loader, trainer):
    data = data_loader()
    trainer(data)

if __name__ == "__main__":
    pipeline_instance = beach_activity_pipeline(
        data_loader=data_loading_step(),
        trainer=training_step()
    )

    pipeline_instance.run()