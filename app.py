import kfp
import kfp.dsl as dsl
from flask import Flask
from flask import request
from pipeline.pipeline import pipeline

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    return "The pipeline service is running", 200


@app.route('/', methods=['POST'])
def run_pipeline():
    hyper_parameters: dict = request.get_json()
    client = kfp.Client(host=hyper_parameters['pipeline_host'])
    client.create_run_from_pipeline_func(
        pipeline,
        arguments={
            'epochs': hyper_parameters['epochs'],
            'learning_rate': hyper_parameters['learning_rate'],
            'batch_size': hyper_parameters['batch_size'],
            'base_model': hyper_parameters['base_model'],
            'last_x_labels': hyper_parameters['last_x_labels'],
            'include_user_labels:': hyper_parameters['include_user_labels']
        },
        mode=dsl.PipelineExecutionMode.V2_COMPATIBLE
    )
    return "New Model training has started", 200


if __name__ == "__main__":
    app.run()
