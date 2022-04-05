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
        'scikit-learn==0.22.1',
        'huggingface-hub',
        'torch',
        'numpy',
        'git+https://github.com/sweng480-team23/tqa-training-lib.git@main'
    ],
    output_component_file="component_config/model_training_component.yaml",
)
def model_training(
        epochs: int,
        learning_rate: str,
        batch_size: int,
        base_model: str,
        train: Input[Dataset],
        val: Input[Dataset],
        model: Output[Model]):
    import pickle

    train_file = open(train.path, 'rb')
    val_file = open(val.path, 'rb')

    train_encodings = pickle.load(train_file)
    val_encodings = pickle.load(val_file)

    from tqa_training_lib.trainers.tf_tweetqa_trainer import TFTweetQATrainer
    from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs

    args = TweetQATrainingArgs(
        epochs=epochs,
        learning_rate=float(learning_rate),
        batch_size=batch_size,
        base_model=base_model,
        model_output_path=model.path,
        use_cuda=True
    )

    trainer = TFTweetQATrainer()
    trainer.train(train_encodings, val_encodings, args)
