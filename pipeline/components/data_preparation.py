from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
)


@component(
    base_image='huggingface/transformers-pytorch-gpu',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'huggingface-hub',
        'torch',
        'fuzzywuzzy',
        'normalise',
        'git+https://github.com/sweng480-team23/tqa-training-lib.git@main',
    ],
    output_component_file="component_config/data_preparation_component.yaml"
)
def data_preparation(data: Input[Dataset], train: Output[Dataset], val: Output[Dataset]):
    import pickle
    import pandas as pd
    df = pd.read_json(data.path)

    from tqa_training_lib.data_preparation_lib import prepare_data
    train_encodings, val_encodings = prepare_data(df, save_data=False, print_stats=False, for_tf=True)

    train_file = open(train.path, 'wb')
    val_file = open(val.path, 'wb')

    pickle.dump(train_encodings, train_file)
    pickle.dump(val_encodings, val_file)
