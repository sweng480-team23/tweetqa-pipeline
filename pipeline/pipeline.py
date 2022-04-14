import kfp.dsl as dsl
from pipeline.components.data_extraction import data_extraction
from pipeline.components.data_preparation import data_preparation
from pipeline.components.model_training import model_training
from pipeline.components.model_scoring import model_scoring


@dsl.pipeline(
    name='TweetQA ML Pipeline',
    description='Machine Learning Pipeline for training of models for the TweetQA System'
)
def pipeline(
        epochs: int,
        learning_rate: str,
        batch_size: int,
        base_model: str,
        last_x_labels: int,
        include_user_labels: bool
):
    data_extraction_task = data_extraction(last_x_labels=last_x_labels, include_user_labels=include_user_labels)
    data_preparation_task = data_preparation(data_extraction_task.outputs['data'])
    model_training_task = model_training(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        base_model=base_model,
        train=data_preparation_task.outputs['train'],
        val=data_preparation_task.outputs['val'])
    model_scoring_task = model_scoring(
        model_loc=model_training_task.outputs['model'],
        cloud_model_url=model_training_task.outputs['output'])
