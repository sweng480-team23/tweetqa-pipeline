from kfp.v2.dsl import (
    component,
    Output,
    Dataset,
)


@component(
    base_image='python:3.8',
    packages_to_install=['pandas==1.3.3'],
    output_component_file="component_config/data_extraction_component.yaml"
)
def data_extraction(url: str, data: Output[Dataset]):
    import pandas as pd

    df = pd.read_json(url)
    df.to_json(data.path)
