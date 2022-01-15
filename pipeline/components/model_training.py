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
    ],
    output_component_file="component_config/model_training_component.yaml",
)
def model_training(train: Input[Dataset], val: Input[Dataset], model: Output[Model], logs: Output[Artifact]):
    import pickle
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from transformers import Trainer
    from transformers import TrainingArguments
    from transformers import BertForQuestionAnswering

    class TweetQADataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    def compute_metrics(p):
        predictions, labels = p
        start_preds = np.argmax(predictions[0], axis=-1)
        end_preds = np.argmax(predictions[1], axis=-1)
        start_labels = labels[0]
        end_labels = labels[1]

        start_accuracy = accuracy_score(y_true=start_labels, y_pred=start_preds)
        start_recall = recall_score(y_true=start_labels, y_pred=start_preds, average='micro')
        start_precision = precision_score(y_true=start_labels, y_pred=start_preds, average='micro')
        start_f1 = f1_score(y_true=start_labels, y_pred=start_preds, average='micro')

        end_accuracy = accuracy_score(y_true=end_labels, y_pred=end_preds)
        end_recall = recall_score(y_true=end_labels, y_pred=end_preds, average='micro')
        end_precision = precision_score(y_true=end_labels, y_pred=end_preds, average='micro')
        end_f1 = f1_score(y_true=end_labels, y_pred=end_preds, average='micro')
        return {
            "start_accuracy": start_accuracy,
            "start_recall": start_recall,
            "start_precison": start_precision,
            "start_f1": start_f1,
            "end_accuracy": end_accuracy,
            "end_recall": end_recall,
            "end_precison": end_precision,
            "end_f1": end_f1
        }

    train_file = open(train.path, 'rb')
    val_file = open(val.path, 'rb')

    train_encodings = pickle.load(train_file)
    val_encodings = pickle.load(val_file)

    train_dataset = TweetQADataset(train_encodings)
    val_dataset = TweetQADataset(val_encodings)

    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    training_args = TrainingArguments(
        output_dir=model.path,
        num_train_epochs=16,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs.path,
        logging_steps=500,
        save_strategy="steps",
        save_steps=500

    )

    bert_model.train()

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(model.path)
