from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact
)


@component(
    base_image='gcr.io/tweetqa-338418/pipeline',
    packages_to_install=[
        'google-cloud-storage',
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
        model: Output[Model]) -> str:
    import pickle

    train_file = open(train.path, 'rb')
    val_file = open(val.path, 'rb')

    train_encodings = pickle.load(train_file)
    val_encodings = pickle.load(val_file)

    from tqa_training_lib.trainers.tf_tweetqa_bert_trainer import TFTweetQABertTrainer
    from tqa_training_lib.trainers.tweetqa_training_args import TweetQATrainingArgs

    args = TweetQATrainingArgs(
        epochs=epochs,
        learning_rate=float(learning_rate),
        batch_size=batch_size,
        base_model=base_model,
        model_output_path=model.path,
        use_cuda=True
    )

    trainer = TFTweetQABertTrainer()
    trainer.train(train_encodings, val_encodings, args)

    # Upload to GCP Storage Bucket
    from google.cloud import storage
    import uuid
    id: str = uuid.uuid4()
    storage_client = storage.Client()
    bucket = storage_client.bucket('tqa-models')
    blob = bucket.blob(f'{id}/tf_model.h5')
    blob.upload_from_filename(f'{model.path}/tf_model.h5')
    blob = bucket.blob(f'{id}/config.json')
    blob.upload_from_filename(f'{model.path}/config.json')
    return f'gs://tqa-models/{id}/'

