from kfp.v2.dsl import (
    component,
    Input,
    Model,
)


@component(
    base_image='gcr.io/tweetqa-338418/pipeline',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'huggingface-hub',
        'torch',
        'numpy',
        'git+https://github.com/Maluuba/nlg-eval.git@master',
    ],
    output_component_file="component_config/model_scoring_component.yaml",
)
def model_scoring(model_loc: Input[Model]):
    import pandas as pd
    import numpy as np
    import string
    import re
    import torch

    from typing import List
    from transformers import BertModel, BertTokenizer
    from nltk.translate.bleu_score import sentence_bleu
    from nlgeval.pycocoevalcap.meteor.meteor import Meteor
    from nlgeval.pycocoevalcap.rouge.rouge import Rouge
    from transformers import BertTokenizerFast, BertForQuestionAnswering

    meteor_scorer = Meteor()
    rouge_scorer = Rouge()

    class BertModelRunner(object):
        tokenizer: BertTokenizer
        model: BertModel

        def __init__(self, tokenizer: BertTokenizer, model: BertModel) -> None:
            self.tokenizer = tokenizer
            self.model = model
            super().__init__()

        def answer_tweet_question(self, tweet, question):
            # tokenize question and text as a pair
            input_ids = self.tokenizer.encode(question, tweet)

            # string version of tokenized ids
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # segment IDs
            # first occurence of [SEP] token
            sep_idx = input_ids.index(self.tokenizer.sep_token_id)
            # number of tokens in segment A (question)
            num_seg_a = sep_idx + 1
            # number of tokens in segment B (text)
            num_seg_b = len(input_ids) - num_seg_a

            # list of 0s and 1s for segment embeddings
            segment_ids = [1] * num_seg_a + [0] * num_seg_b
            assert len(segment_ids) == len(input_ids)

            # model output using input_ids and segment_ids
            self.model.eval()
            output = self.model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([segment_ids]))

            # reconstructing the answer
            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits)
            if answer_end >= answer_start:
                answer = tokens[answer_start]
                for i in range(answer_start + 1, answer_end + 1):
                    if tokens[i][0:2] == "##":
                        answer += tokens[i][2:]
                    else:
                        answer += " " + tokens[i]
            else:
                answer = "Unable to find the answer to your question."

            return answer, answer_start.item(), answer_end.item()

    def read_data(url: str) -> pd.DataFrame:
        return pd.read_json(url)

    def get_model_runner() -> BertModelRunner:
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        model = BertForQuestionAnswering.from_pretrained(model_loc.path)
        return BertModelRunner(tokenizer, model)

    def generate_gold_file(df: pd.DataFrame) -> List[dict]:
        data_dict: dict = df.to_dict('records')
        return [{'qid': datum['qid'], 'Answer': datum['Answer']} for datum in data_dict]

    def to_prediction(datum: dict, model_runner: BertModelRunner) -> dict:
        answer, start, end = model_runner.answer_tweet_question(datum['Tweet'], datum['Question'])
        return {
            'qid': datum['qid'],
            'Tweet': datum['Tweet'],
            'Question': datum['Question'],
            'Answer': answer,
            'Start': start,
            'End': end,
            'Actual Answer': datum['Answer']
        }

    def generate_user_file(df: pd.DataFrame) -> List[dict]:
        model_runner: BertModelRunner = get_model_runner()
        data_dict: dict = df.to_dict('records')
        return [to_prediction(datum, model_runner) for datum in data_dict]

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

    def ans_score(ans, gold_list):
        ans = normalize_answer(ans)
        gold_list = [normalize_answer(ref) for ref in gold_list]
        bleu = sentence_bleu([_.split() for _ in gold_list], ans.split(), weights=(1, 0, 0, 0))
        meteor, _ = meteor_scorer.compute_score({0: gold_list}, {0: [ans]})
        rouge, _ = rouge_scorer.compute_score({0: gold_list}, {0: [ans]})
        return {'bleu': bleu, 'meteor': meteor, 'rouge': rouge}

    def evaluate(gold, pred):
        idx2gold = {item['qid']: item['Answer'] for item in gold}
        idx2pred = {item['qid']: item['Answer'] for item in pred}
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

        return {
            'BLEU-1': np.mean(bleus),
            'METEOR': np.mean(meteors),
            'ROUGE': np.mean(rouges)
        }

    data: pd.DataFrame = read_data('https://raw.githubusercontent.com/sweng480-team23/tweet-qa-data/main/dev.json')
    gold_file = generate_gold_file(data)
    user_file = generate_user_file(data)
    scores = evaluate(gold_file, user_file)
    print(scores)
    # TODO: get existing model, update score values, save updates
    # model.bleu_score = scores['BLEU-1']
    # model.meteor_score = scores['METEOR']
    # model.rouge_score = scores['ROUGE']
