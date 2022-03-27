from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact
)


@component(
    base_image='huggingface/transformers-pytorch-gpu',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'huggingface-hub',
        'torch',
        'numpy',
        'git+https://github.com/sweng480-team23/tqa-training-lib.git@main'
    ],
    output_component_file="component_config/model_training_component.yaml",
)
def model_training(train: Input[Dataset], val: Input[Dataset], model: Output[Model], logs: Output[Artifact]):
    import pickle

    train_file = open(train.path, 'rb')
    val_file = open(val.path, 'rb')

    train_encodings = pickle.load(train_file)
    val_encodings = pickle.load(val_file)

    from tqa_training_lib.model_training_lib import do_train
    do_train(train_encodings, val_encodings, model.path, logs.path)
