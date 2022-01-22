import kfp
import kfp.dsl as dsl
from components.data_extraction import data_extraction
from components.data_preparation import data_preparation
from components.model_training import model_training

client = kfp.Client(host='https://79f549af1655b99d-dot-us-central1.pipelines.googleusercontent.com/')


@dsl.pipeline(
    name='TweetQA ML Pipeline',
    description='Machine Learning Pipeline for training of models for the TweetQA System'
)
def pipeline(url: str):
    data_extraction_task = data_extraction(url=url)
    data_preparation_task = data_preparation(data_extraction_task.outputs['data'])
    model_training_task = model_training(data_preparation_task.outputs['train'], data_preparation_task.outputs['val'])


client.create_run_from_pipeline_func(
    pipeline,
    arguments={
        'url': 'https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json',
    },
    mode=dsl.PipelineExecutionMode.V2_COMPATIBLE
)
