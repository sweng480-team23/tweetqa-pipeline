import kfp
import kfp.dsl as dsl
from components.data_extraction import data_extraction
from components.add import add


client = kfp.Client(host='https://483b022f2ba62734-dot-us-central1.pipelines.googleusercontent.com')


@dsl.pipeline(
    name='TweetQA ML Pipeline',
    description='Machine Learning Pipeline for training of models for the TweetQA System'
)
def pipeline(url: str, a: float, b: float):
    add_task = add(a, b)
    data_extraction_task = data_extraction(url=url)


client.create_run_from_pipeline_func(
    pipeline,
    arguments={
        'url': 'https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/train.json',
        'a': 1,
        'b': 16
    },
    mode=dsl.PipelineExecutionMode.V2_COMPATIBLE
)
