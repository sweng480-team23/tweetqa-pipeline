from kfp.v2.dsl import (
    component,
    Output,
    Dataset,
)


@component(
    base_image='python:3.8',
    packages_to_install=[
        'pandas==1.3.3',
        'git+https://github.com/sweng480-team23/tqa-training-lib.git@main'
    ],
    output_component_file="component_config/data_extraction_component.yaml"
)
def data_extraction(last_x_labels: int, include_user_labels: bool, data: Output[Dataset]):
    from tqa_training_lib.data_extraction_lib import extract_data

    extract_data(last_x_labels, include_user_labels).to_json(data.path)
