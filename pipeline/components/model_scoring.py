from kfp.v2.dsl import (
    component,
    Input,
    Model,
)


@component(
    base_image='gcr.io/tweetqa-338418/pipeline',
    packages_to_install=[
        'pandas',
        'scikit-learn==0.22.1',
        'huggingface-hub',
        'torch',
        'numpy',
        'git+https://github.com/Maluuba/nlg-eval.git@master',
        'git+https://github.com/sweng480-team23/tqa-training-lib.git@main'
    ],
    output_component_file="component_config/model_scoring_component.yaml",
)
def model_scoring(model_loc: Input[Model], cloud_model_url: str):
    from tqa_training_lib.model_scoring_lib import score_model
    import nltk
    import requests
    nltk.download('brown')
    nltk.download('names')
    scores = score_model(model_loc.path, save_gold_user_files=True, print_scores=True, use_tf=True)
    new_model: dict = {
        "visitor": {
            "id": 3,
            "token": "30fc03a7-8fe8-45ab-9c05-e5762f0c480b"
        },
        "ml_type": "BERT",
        "ml_version": "1.0.0",
        "bleu_score": scores['BLEU-1'],
        "rouge_score": scores['ROUGE'],
        "meteor_score": scores['METEOR'],
        "model_url": cloud_model_url
    }
    requests.post(url='https://tweetqa-api-d62rdgteaa-uc.a.run.app/v2/models', json=new_model)
