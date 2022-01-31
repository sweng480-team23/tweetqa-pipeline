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
        'fuzzywuzzy'
    ],
    output_component_file="component_config/data_preparation_component.yaml"
)
def data_preparation(data: Input[Dataset], train: Output[Dataset], val: Output[Dataset]):
    import pandas as pd
    
    from typing import Tuple
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizerFast
    from fuzzywuzzy import fuzz, process
    import pickle

    def lower_case_filter(datum: dict):
        answer = datum["Answer"].lower()
        tweet = datum["Tweet"].lower()
        question = datum["Question"].lower()
        
        return {"Answer": answer,
                "qid": datum["qid"],
                "Question": question,
                "Tweet": tweet}

    def fuzzy_match(tweet: str, answer: str) -> Tuple[int, int]:
        canidates = []
        tweet_split = tweet.split()
        answer_split = answer.split()
        
        n = len(answer_split)
        m = len(tweet_split)
        
        for i in range(m - n):
            canidates.append(tweet_split[i:i+n])
            
        canidates = [' '.join(canidate) for canidate in canidates] 
        
        best_matches = process.extractBests(answer,
                                            canidates,
                                            scorer=fuzz.token_sort_ratio,
                                            score_cutoff=75)
        
        if best_matches:
            best_matches = [(match[1], match[0]) for match in best_matches]
            best_match = max(best_matches)[1]
            
            start_position = tweet.find(best_match)
            end_position = start_position + len(best_match)
            
            return start_position, end_position
        else:
            return -1, -1
        
    def identify_start_and_end_positions(datum: dict) -> dict:
        tweet = datum["Tweet"]
        question = datum["Question"]
        answer = datum["Answer"]
        
        start_position = tweet.find(answer)
        
        if start_position > -1:
            end_position = start_position + len(answer)
        else:
            start_position, end_position = fuzzy_match(tweet, answer)
            
            
        assert start_position <= end_position, f'{start_position} > {end_position}'
        
        return {
            "qid": datum["qid"],
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
    x_train = train_data.to_dict('records')
    x_val = val_data.to_dict('records')


    # List of functions we want to call on the data 
    filters = [lower_case_filter,
            identify_start_and_end_positions]

    for f in filters:
        x_train = [f(datum) for datum in x_train]
        x_val = [f(datum) for datum in x_val]

    non_quality_x_train = [datum for datum in x_train if datum["start_position"] == -1]
    non_quality_x_val = [datum for datum in x_val if datum["start_position"] == -1]
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
