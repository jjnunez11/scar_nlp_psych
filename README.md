# Predicting Which Patients with Cancer Patients will see a Psychiatrist or Counsellor from their Initial Oncology Consultation Document Using Natural Language Processing

By John-Jose Nunez, on behalf of the co-authors

For the forthcoming paper:  [Title](https://url)

TODO: UPDATE BELOW FROM SURVIVAL PAPER

Implementation of NLP methods to predict clinical-addressed psychosocail cancer needs using physician documents.
Most of the data processing and selection was done in the scar_nlp_data and scar repos, also located on my Github. 

Loosely based on the hedwig NLP repo, but starting fresh due to the large changes
in PyTorch 1.9.0 and especially torchtext 0.10.0. With some updated merged in from related working predicting 
survival using this dataset, located in my repo jjnunez\scar_nlp_survival

# Training/Fine-tuning NLP Models

Models are ran as python modules. Arguments can be based through the command line. E.g.

python -m cnn --target "dspln_PSYCHIATRY_12" --batch-size 16

See [models](./models) for the various deployed models. 

See [trainers](./trainers) for the trainers used to train the models

See [evaluators](./evaluators) for the code used to evaluate models

See [results](./results) for the final results and trained models

See [tables](./results) for some code used to generate tables as the raw data used for tables

For training the models on Windows, .bat files are provided that contain the command-line code,
as found in the [bats](./bats) folder. Sorry, I had to use Windows, as IT didn't support Linux on their virtualization.

To use BERT, please first download bert-base-uncased from [HuggingFace](https://huggingface.co/bert-base-uncased/tree/main)
The pytorch_model, config, and vocab files should be placed in a folder named bert-base-uncased, whose directory
in provided in the argument/default argument. 

# Visualizing and Understanding Models

See the [viz](./viz) folder for jupyter notebooks used to visualize and understand the models.

Thank you for your interest in our work! Please don't hesitate to reach out - John-Jose, johnjose.nunez@bccancer.bc.ca





