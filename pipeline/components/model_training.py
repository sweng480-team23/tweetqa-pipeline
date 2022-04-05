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
def model_training(train: Input[Dataset], val: Input[Dataset], model: Output[Model]):
    import pickle

    train_file = open(train.path, 'rb')
    val_file = open(val.path, 'rb')

    train_encodings = pickle.load(train_file)
    val_encodings = pickle.load(val_file)

    from tqa_training_lib.trainers.tf_tweetqa_trainer import TFTweetQATrainer
    from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs

    args = TweetQATrainingArgs(
        epochs=2,
        learning_rate=2.9e-5,
        batch_size=8,
        base_model='bert-large-uncased-whole-word-masking-finetuned-squad',
        model_output_path=model.path,
        use_cuda=True
    )

    trainer = TFTweetQATrainer()
    trainer.train(train_encodings, val_encodings, args)
