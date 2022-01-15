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
        'torch'
    ],
    output_component_file="component_config/data_preparation_component.yaml"
)
def data_preparation(data: Input[Dataset], train: Output[Dataset], val: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizerFast
    import pickle

    def identify_start_and_end_positions(instance: dict) -> dict:
        tweet = instance["Tweet"].lower()
        question = instance["Question"].lower()
        answer = instance["Answer"].lower()
        start_position = tweet.find(answer)

        if start_position > -1:
            end_position = start_position + len(answer)
        else:
            end_position = -1

        return {
            "qid": instance["qid"],
            "tweet": tweet,
            "question": question,
            "answer": answer,
            "start_position": start_position,
            "end_position": end_position,
        }

    def add_token_positions(encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['start_position']))
            end_positions.append(encodings.char_to_token(i, answers[i]['end_position'] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    df = pd.read_json(data.path)
    df["Answer"] = df["Answer"].explode()
    train_data, val_data = train_test_split(df, test_size=0.2)
    x_train_records = train_data.to_dict('records')
    x_val_records = val_data.to_dict('records')

    x_train = [identify_start_and_end_positions(datum) for datum in x_train_records]
    x_val = [identify_start_and_end_positions(datum) for datum in x_val_records]

    quality_x_train = [datum for datum in x_train if datum["start_position"] >= 0]
    quality_x_val = [datum for datum in x_val if datum["start_position"] >= 0]

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    train_encodings = tokenizer(
        [q["tweet"] for q in quality_x_train],
        [q["question"] for q in quality_x_train],
        max_length=50,
        padding='max_length',
        truncation=True)

    val_encodings = tokenizer(
        [q["tweet"] for q in quality_x_val],
        [q["question"] for q in quality_x_val],
        max_length=50,
        padding='max_length',
        truncation=True)

    add_token_positions(train_encodings, quality_x_train)
    add_token_positions(val_encodings, quality_x_val)

    train_file = open(train.path, 'wb')
    val_file = open(val.path, 'wb')

    pickle.dump(train_encodings, train_file)
    pickle.dump(val_encodings, val_file)
