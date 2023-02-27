:: tbl_see_psych
:: Performance predicting whether a patient will see a psychiatrist in first 1 year (12 months)
:: Repeat x10 for variance estimation

:: BOW
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --classifier "l2logreg" --l2logreg-c 0.6 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --classifier "l2logreg" --l2logreg-c 0.6 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --classifier "l2logreg" --l2logreg-c 0.6 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --classifier "l2logreg" --l2logreg-c 0.6 --epochs 1 --max-tokens 5000
python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --classifier "l2logreg" --l2logreg-c 0.6 --epochs 1 --max-tokens 5000

:: CNN
python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "see_psych"  --weight-decay 0.0001 --dropout 0.85 --lr 0.0001
python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "see_psych"  --weight-decay 0.0001 --dropout 0.85 --lr 0.0001
python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "see_psych"  --weight-decay 0.0001 --dropout 0.85 --lr 0.0001
python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "see_psych"  --weight-decay 0.0001 --dropout 0.85 --lr 0.0001
python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "see_psych"  --weight-decay 0.0001 --dropout 0.85 --lr 0.0001

:: LSTM
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "see_psych" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "see_psych" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "see_psych" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "see_psych" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "see_psych" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001

:: BERT
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "see_psych" --weight-decay 0 --lr 0.001
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "see_psych" --weight-decay 0 --lr 0.001
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "see_psych" --weight-decay 0 --lr 0.001
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "see_psych" --weight-decay 0 --lr 0.001
python -m models.bert --target "dspln_PSYCHIATRY_12" --table "see_psych" --weight-decay 0 --lr 0.001

