import os
from copy import deepcopy
from trainers.rule_trainer import RuleTrainer
from evaluators.evaluator import Evaluator
from models.bow.args import get_args
from datasets.scar_bow import SCARBoW
import warnings
import datetime

if __name__ == '__main__':
    args = get_args()

    # Setup config
    config = deepcopy(args)
    model_name = "Rule"
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    config.run_name = model_name + "_" + start_time

    print(f"Evaluating a {model_name} model")

    scar_bow = SCARBoW(args, config.eval_only)  # Uses same data and processesing as BoW

    # Find feature index for use with this rule
    if "PSYCHIATRY" in config.target:
        rule_token = "psychiatrist"
    elif "SOCIALWORK" in config.target:
        rule_token = "counsel"
    else:
        raise ValueError(f"Could not use rule-based method with this target: {config.target}")

    rule_token_idx = scar_bow.get_idx_of_token(rule_token)
    print(f"Rule is based on presence of token: {rule_token} at index {rule_token_idx}")

    # Make directories for results if not already there
    config.results_dir_target = os.path.join(config.results_dir, config.target)  # dir for a targets results
    config.results_dir_model = os.path.join(config.results_dir_target, model_name)  # subdir for each model

    if not os.path.exists(config.results_dir_target):
        os.mkdir(config.results_dir_target)
    if not os.path.exists(config.results_dir_model):
        os.mkdir(config.results_dir_model)

    # Train and Evaluate Model
    trainer = RuleTrainer(rule_token_idx=rule_token_idx)
    train_data = scar_bow.get_train_data()
    dev_data = scar_bow.get_dev_data()
    test_data = scar_bow.get_test_data()
    train_history, dev_history, test_history, start_time = trainer.fit(train_data, dev_data, test_data,
                                                                           config.epochs)
    evaluator = Evaluator("Rule", test_history, config, start_time)

    # Use evaluator to print the best epochs
    print('\nBest epoch for AUC:')
    evaluator.print_best_auc()

    print('\nBest epoch for F1:')
    evaluator.print_best_f1()

    # Write the run history, and update the master results file
    evaluator.write_result_history()
    evaluator.append_to_results()
