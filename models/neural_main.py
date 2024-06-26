import os
import sys
from copy import deepcopy
import torch.nn as nn
import torch
from evaluators.evaluator import Evaluator
from datasets.scar import SCAR
import datetime

from tables.generate_token_counts import count_neural_tokens


def neural_main(model_name, model_class, model_trainer, args):
    """
    Moved the __main__ code to this function, given the shared code for the implementation of our CNN and LSTM models

    :param model_name: string name
    :param model_class: pytorch model
    :param model_trainer: train object depending on which model is used
    :param args: arguments passed in by the get_args function of the args file of that modal
    :return: nothing, but loads, trains, and evaluates the models
    """
    eval_only = args.eval_only
    if eval_only:
        print(f"Loading and evaluating a {model_name} model")
    else:
        print(f"Training and evaluating a {model_name} model")

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Set CUDA Blocking if needed, used when getting CUDA errors:
    if args.cuda_block:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

    # Set device being used to train and evaluate the model
    if args.cuda:
        args.device = torch.device('cuda:0')
        print("Using a CUDA GPU, woot!")
    else:
        args.device = 'cpu'
        print("Using a CPU, sad!")

    # Loss and optimizer

    if eval_only:
        loss_fn = nn.BCEWithLogitsLoss()  # Does not matter, we're only evaluating
        scar = SCAR(args.batch_size, args.data_dir, args.target, eval_only=eval_only)
    elif args.imbalance_fix == 'loss_weight':
        scar = SCAR(args.batch_size, args.data_dir, args.target, eval_only=eval_only)
        target_perc = scar.get_class_balance()  # Percentage of targets = 1
        pos_weight = (1 - target_perc) / target_perc
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    elif args.imbalance_fix == 'none':
        loss_fn = nn.BCEWithLogitsLoss()
        scar = SCAR(args.batch_size, args.data_dir, args.target, eval_only=eval_only)
    elif args.imbalance_fix == 'undersampling':
        loss_fn = nn.BCEWithLogitsLoss()
        scar = SCAR(args.batch_size, args.data_dir, args.target, eval_only=eval_only, undersample=True)
    else:
        raise Exception("Invalid method to fix the class imbalance provided, or not yet implemented")

    # Setup config
    config = deepcopy(args)
    config.target_classes = scar.NUM_CLASSES
    config.vocab_size = scar.get_vocab_size()
    config.run_name = model_name + "_" + start_time

    # Make directories for results if not already there
    config.results_dir_target = os.path.join(config.results_dir, config.target)  # dir for a targets results
    config.results_dir_model = os.path.join(config.results_dir_target, model_name)  # subdir for each model

    if not os.path.exists(config.results_dir_target):
        os.mkdir(config.results_dir_target)
    if not os.path.exists(config.results_dir_model):
        os.mkdir(config.results_dir_model)

    # Instantiate our Model
    model = model_class(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    trainer = model_trainer(model, optimizer, loss_fn, config, args.imbalance_fix)

    # Train and Evaluate Model
    if eval_only:
        test_dataloader = scar.test_dataloader()
        test_history, start = trainer.eval_only(model_class, test_dataloader)
    elif config.count_tokens: # If we're just running this to count tokens in our documents, exit this script and call relevant script
        train_dataloader = scar.train_dataloader()
        dev_dataloader = scar.dev_dataloader()
        test_dataloader = scar.test_dataloader()
        count_neural_tokens(model_name, config, train_dataloader, dev_dataloader, test_dataloader)
        sys.exit()
    else:
        train_dataloader = scar.train_dataloader()
        dev_dataloader = scar.dev_dataloader()
        test_dataloader = scar.test_dataloader()
        train_history, dev_history, test_history, start_time = trainer.fit(train_dataloader,
                                                                           dev_dataloader,
                                                                           test_dataloader)
    evaluator = Evaluator(model_name, test_history, config, start_time)

    # Write the run history, and update the master results file
    evaluator.write_result_history()
    evaluator.append_to_results()
