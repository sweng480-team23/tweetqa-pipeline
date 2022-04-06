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
def model_scoring(model_loc: Input[Model]):
    import nltk
    nltk.download('brown')
    from tqa_training_lib.model_scoring_lib import score_model
    score_model(model_loc.path, save_gold_user_files=True, print_scores=True, use_tf=True)
