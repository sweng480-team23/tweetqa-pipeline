from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact
)

# TODO - what base image to use?
@component(
    base_image='lwestfall/tqa-scorer',
    packages_to_install=[
        'pandas',
        'numpy',
        'huggingface-hub',
        'torch',
        'git+https://github.com/Maluuba/nlg-eval.git@master',
        'nltk',
    ],
    output_component_file="component_config/scoring_component.yaml",
)

def scoring(val: Input[Dataset], model: Input[Model], scores: Output[Model]):
    import pickle
    import pandas
    import json
    import string
    import re
    import torch

    # import nltk
    # nltk.download()

    from nltk.translate.bleu_score import sentence_bleu
    import numpy as np

    from nlgeval.pycocoevalcap.meteor.meteor import Meteor
    from nlgeval.pycocoevalcap.rouge.rouge import Rouge
    from transformers import BertTokenizerFast
    from transformers import BertForQuestionAnswering

    class TweetQADataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    # get an answer from the model
    # adapted from colab
    def answer_tweet_question(bert_model, tweet, question):
        #tokenize question and text as a pair
        input_ids = tokenizer.encode(question, tweet)

        #string version of tokenized ids
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        #segment IDs
        #first occurence of [SEP] token
        sep_idx = input_ids.index(tokenizer.sep_token_id)
        #number of tokens in segment A (question)
        num_seg_a = sep_idx+1
        #number of tokens in segment B (text)
        num_seg_b = len(input_ids) - num_seg_a

        #list of 0s and 1s for segment embeddings
        segment_ids = [1]*num_seg_a + [0]*num_seg_b
        assert len(segment_ids) == len(input_ids)

        #model output using input_ids and segment_ids
        # output = bert_model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([segment_ids]).to(device))
        bert_model.eval()
        output = bert_model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([segment_ids]))

        #reconstructing the answer
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        print(f"Answer Start: {answer_start}")
        print(f"Answer End: {answer_end}")
        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start+1, answer_end+1):
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]

        # if answer.startswith("[CLS]"):
        #     answer = "Unable to find the answer to your question."

        # print("\nPredicted answer:\n{}".format(answer.capitalize()))

        return answer

    #### pulled directly from tweetqa_eval.py
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    meteor_scorer = Meteor()
    rouge_scorer = Rouge()

    #### pulled directly from tweetqa_eval.py
    def ans_score(ans, gold_list):
        ans = normalize_answer(ans)
        gold_list = [normalize_answer(ref) for ref in gold_list]
        bleu = sentence_bleu([_.split() for _ in gold_list], ans.split(), weights=(1,0,0,0))
        meteor, _ = meteor_scorer.compute_score({0:gold_list}, {0:[ans]})
        rouge, _ = rouge_scorer.compute_score({0:gold_list}, {0:[ans]})
        return {'bleu': bleu, 'meteor':meteor, 'rouge': rouge}

    #### pulled directly from tweetqa_eval.py
    def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
        gold_file = test_annotation_file
        pred_file = user_annotation_file
        gold = json.load(open(gold_file))
        pred = json.load(open(pred_file))
        idx2gold = {item['qid']:item['Answer'] for item in gold}
        idx2pred = {item['qid']:item['Answer'] for item in pred}
        idx2scores = {}
        for id_ in idx2gold.keys():
            if isinstance(idx2pred[id_], list):
                pred_ans = idx2pred[id_][0]
            else:
                pred_ans = idx2pred[id_]
            idx2scores[id_] = ans_score(pred_ans, idx2gold[id_])
        bleus = [item['bleu'] for item in idx2scores.values()]
        meteors = [item['meteor'] for item in idx2scores.values()]
        rouges = [item['rouge'] for item in idx2scores.values()]
        print({'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

        output = {}
        output['result'] = [
        {'test_split':
            {
            'BLEU-1': np.mean(bleus),
            'METEOR': np.mean(meteors),
            'ROUGE': np.mean(rouges)
            }
        }
        ]

        return output

    # get the tokenizer, might need to switch to the trained model?? not sure
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # open binary serialization of validation set from model_training
    val_file = open(val.path, 'rb')

    # deserialize
    val_encodings = pickle.load(val_file)

    # get dataset
    pred_list = val_dataset(val_encodings)

    # load up the trained model
    bert_model = BertForQuestionAnswering.from_pretrained(model.path)

    # print(json.dumps(pred_list))

    # have new model answer questions
    for tqa in pred_list:
        print("hello")
        print(json.dumps(tqa))
        tqa.Answer = answer_tweet_question(bert_model, tqa.Tweet, tqa.Question)
        # tqa["Answer"] = answer_tweet_question(bert_model, tqa["Tweet"], tqa["Question"])
        # ans = answer_tweet_question(bert_model, tqa["Tweet"], tqa["Question"])
        # if not ans.startswith("[CLS]"):
        #     tqa["Answer"] = ans

    # save to JSON files for evaluate function
    with open('user_annotations.json', 'w') as f:
        json.dump(val_encodings, f)
    with open('pred_annotations.json', 'w') as f:
        json.dump(pred_list, f)

    # get scores
    scores = evaluate('pred_annotations.json', 'user_annotations.json')

    # save scores to file
    with open('scores_out.json', 'w') as f:
        json.dump(scores, f)
