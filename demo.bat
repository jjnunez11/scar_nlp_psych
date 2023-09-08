:: Evaluating previously trained models to predict seeing a psychiatrist within 12 months using toy data
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "demo" --eval_only --model-file ".\results\final_results\dspln_PSYCHIATRY_12\BoW\BoW_20230220-1206_e0.pbz2" --data-dir ".\demo" --results-dir ".\demo"
python -m models.cnn  --target "dspln_PSYCHIATRY_12" --table "demo"  --eval_only --model-file ".\results\final_results\dspln_PSYCHIATRY_12\CNN\CNN_20230214-0927.pt" --data-dir ".\demo" --results-dir ".\demo"
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "demo"  --eval_only --model-file ".\results\final_results\dspln_PSYCHIATRY_12\LSTM\LSTM_20230214-1515.pt" --data-dir ".\demo" --results-dir ".\demo"
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "demo" --eval_only --model-file ".\results\final_results\dspln_PSYCHIATRY_12\BERT\default\version_2\BERT--epoch=4_val_bal_val_bal=0.72.ckpt" --data-dir ".\demo" --results-dir ".\demo" --pretrained_dir ".\demo"

:: Evaluating previously trained models to predict seeing a counsellor within 12 months using toy data
python -m models.bow --target "dspln_SOCIALWORK_12" --table "demo" --eval_only --model-file ".\results\final_results\dspln_SOCIALWORK_12\BoW\BoW_20230220-1316_e0.pbz2" --data-dir ".\demo" --results-dir ".\demo"
python -m models.cnn  --target "dspln_SOCIALWORK_12" --table "demo"  --eval_only --model-file ".\results\final_results\dspln_SOCIALWORK_12\CNN\CNN_20230216-1157.pt" --data-dir ".\demo" --results-dir ".\demo"
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "demo"  --eval_only --model-file ".\results\final_results\dspln_SOCIALWORK_12\LSTM\LSTM_20230216-1518.pt" --data-dir ".\demo" --results-dir ".\demo"
python -m models.bert --target "dspln_SOCIALWORK_12" --table "demo" --eval_only --model-file ".\results\final_results\dspln_SOCIALWORK_12\BERT\default\version_0\BERT--epoch=20_val_bal_val_bal=0.66.ckpt" --data-dir ".\demo" --results-dir ".\demo" --pretrained_dir ".\demo"

:: Load toy sentences and run our multi-document interpretation
python -m multiligtopic --load_sents --load_file ".\demo\multiligtopic\toy_impt_sents.txt"