:: tbl_see_counselling
:: Performance predicting whether a patient will see a counsellor (SOCIALWORK) in first 1 year (12 months)
:: Repeat x5 for variance estimation

:: BOW
python -m models.bow --target "dspln_SOCIALWORK_12" --table "see_counselling" --classifier "l2logreg" --l2logreg-c 1.05 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_SOCIALWORK_12" --table "see_counselling" --classifier "l2logreg" --l2logreg-c 1.05 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_SOCIALWORK_12" --table "see_counselling" --classifier "l2logreg" --l2logreg-c 1.05 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_SOCIALWORK_12" --table "see_counselling" --classifier "l2logreg" --l2logreg-c 1.05 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_SOCIALWORK_12" --table "see_counselling" --classifier "l2logreg" --l2logreg-c 1.05 --epochs 1 --max-tokens 5000

:: CNN
python -m models.cnn --target "dspln_SOCIALWORK_12" --table "see_counselling"  --weight-decay 0 --dropout 0.85 --lr 0.00005
python -m models.cnn --target "dspln_SOCIALWORK_12" --table "see_counselling"  --weight-decay 0 --dropout 0.85 --lr 0.00005
python -m models.cnn --target "dspln_SOCIALWORK_12" --table "see_counselling"  --weight-decay 0 --dropout 0.85 --lr 0.00005
python -m models.cnn --target "dspln_SOCIALWORK_12" --table "see_counselling"  --weight-decay 0 --dropout 0.85 --lr 0.00005
python -m models.cnn --target "dspln_SOCIALWORK_12" --table "see_counselling"  --weight-decay 0 --dropout 0.85 --lr 0.00005

:: LSTM
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "see_counselling" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "see_counselling" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "see_counselling" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "see_counselling" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
python -m models.lstm --target "dspln_SOCIALWORK_12" --table "see_counselling" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005

:: BERT
python -m models.bert --target "dspln_SOCIALWORK_12" --table "see_counselling" --weight-decay 0.01 --lr 0.00005
python -m models.bert --target "dspln_SOCIALWORK_12" --table "see_counselling" --weight-decay 0.01 --lr 0.00005
python -m models.bert --target "dspln_SOCIALWORK_12" --table "see_counselling" --weight-decay 0.01 --lr 0.00005
python -m models.bert --target "dspln_SOCIALWORK_12" --table "see_counselling" --weight-decay 0.01 --lr 0.00005
python -m models.bert --target "dspln_SOCIALWORK_12" --table "see_counselling" --weight-decay 0.01 --lr 0.00005


