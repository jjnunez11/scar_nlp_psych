:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --patience 10
:: python -m models.cnn --lr 0.0001 --target "dspln_PSYCHIATRY_0"
:: python -m models.cnn --lr 0.000001 --target "dspln_PSYCHIATRY_0"
:: python -m models.cnn --lr 0.0000001 --target "dspln_PSYCHIATRY_0"

:: python -m models.lstm --lr 0.00001 --target "dspln_PSYCHIATRY_0" --epochs 1
:: python -m models.cnn --lr 0.00001 --target "surv_mo_60" --patience 10 --epochs 100

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --table "see_psych" --epochs 1

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --table "see_psych"
:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_12" --table "see_psych"


:: python -m models.cnn --lr 0.000001 --target "dspln_PSYCHIATRY_60" --table "see_psych" --patience 10 --epochs 1000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "rf"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "elnet"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "gbdt"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5

:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 500
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 1000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 2000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 10000

:: python -m models.bow --target "dspln_PSYCHIATRY_60" --epochs 1 --classifier "l2logreg" --max-tokens 100 --l2logreg-c 1 --table "test"
:: python -m models.cnn --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 16
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 64
:: python -m models.bert --target "dsplnic_PSYCHIATRY_60" --table "test" --epochs 2

:: python -m models.bow --target "survic_mo_60" --epochs 1
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --epochs 1
:: python -m models.bow --target "dsplnic_SOCIALWORK_60" --epochs 1
:: python -m models.bow --target "need_emots_4" --epochs 1

:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 64 --epochs 1
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 32 --epochs 1

:: Re-run LSTM models after fixing to match prior dsplnic_SOCIALWORK_60
:: python -m models.lstm --target "need_emots_4" --table "need_emots_all_models"
:: python -m models.lstm --target "need_infos_4" --table "need_infos_all_models"
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "see_psych"
:: python -m models.lstm --target "dsplnic_SOCIALWORK_60" --table "see_counselling"
:: python -m models.lstm --target "survic_mo_60" --table "survival_all_models"

:: python -m models.cnn --target "dsplnic_PSYCHIATRY_60"

:: python -m models.bow --target "dspln_PSYCHIATRY_60" --epochs 1
:: python -m models.cnn --target "dspln_PSYCHIATRY_60" --epochs 1

:: python -m models.cnn --target dspln_PSYCHIATRY_60 --table cnn_tuning_post_spec_fix --lr 0.0001 --dropout 0.8 --weight-decay 0.0001
:: python -m models.cnn --target dspln_PSYCHIATRY_60 --table cnn_tuning_post_spec_fix --lr 0.0001 --dropout 0.8 --weight-decay 0.0001
:: python -m models.cnn --target dspln_PSYCHIATRY_60 --table cnn_tuning_post_spec_fix --lr 0.0001 --dropout 0.8 --weight-decay 0.0001
:: python -m models.cnn --target dspln_PSYCHIATRY_60 --table cnn_tuning_post_spec_fix --lr 0.0001 --dropout 0.8 --weight-decay 0.0001

:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.99 --imbalance-fix undersampling
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 1 --imbalance-fix undersampling
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0 --imbalance-fix undersampling

:: call .\bats\tables\see_psych.bat
:: call .\bats\tables\see_counselling.bat
:: call .\bats\tables\compare_sexes.bat

:: tuning best few
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.3 --wdrop 0 --embed-droprate 0.001 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.1 --wdrop 0 --embed-droprate 0.1 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.1 --wdrop 0.01 --embed-droprate 0.1 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.3 --wdrop 0.0001 --embed-droprate 0.01 --lr 0.0001

:: python -m models.lstm --target dspln_PSYCHIATRY_60 --table "lstm_tuning" --dropout 0.2 --wdrop 0.0001 --embed-droprate 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60 --table "lstm_tuning"  --dropout 0.3 --wdrop 0.0001 --embed-droprate 0.1 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60 --table "lstm_tuning"  --dropout 0.1 --wdrop 0.001 --embed-droprate 00.1 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60 --table "lstm_tuning"  --dropout 0.1 --wdrop 0.0001 --embed-droprate 0.1 --lr 0.0005

:: python -m models.bert --target "dspln_PSYCHIATRY_60_toy" --table "test" --epochs 1
:: python -m models.cnn --target "dspln_PSYCHIATRY_60_toy" --table "test" --epochs 1
:: python -m models.lstm --target "dspln_PSYCHIATRY_60_toy" --table "test" --epochs 1

:: Can run these for a bit more tuning
:: python .\bats\tuning\cnn_tuning.py
:: python .\bats\tuning\cnn_tuning.py

:: python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "chg_durations"
:: python -m models.cnn --target "dspln_PSYCHIATRY_12" --table "chg_durations"
:: python -m models.cnn --target "dspln_PSYCHIATRY_24" --table "chg_durations"
:: python -m models.cnn --target "dspln_PSYCHIATRY_24" --table "chg_durations"
:: python -m models.cnn --target "dspln_PSYCHIATRY_36" --table "chg_durations"
:: python -m models.cnn --target "dspln_PSYCHIATRY_36" --table "chg_durations"

:: python .\bats\tuning\cnn_tuning.py
:: python .\bats\tuning\cnn_tuning_sw.py

:: python -m models.cnn --target "dspln_SOCIALWORK_12" --table "cnn_tuning"
:: python -m models.cnn --target "dspln_SOCIALWORK_12" --table "cnn_tuning"
:: python -m models.cnn --target "dspln_SOCIALWORK_24" --table "cnn_tuning"
:: python -m models.cnn --target "dspln_SOCIALWORK_24" --table "cnn_tuning"
:: python -m models.cnn --target "dspln_SOCIALWORK_36" --table "cnn_tuning"
:: python -m models.cnn --target "dspln_SOCIALWORK_36" --table "cnn_tuning"

:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning"
:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" '--lr' '--dropout' '--weight-decay' '--embed-droprate' '--wdrop'

:: python -m models.bow --target "dspln_PSYCHIATRY_12" --table "see_psych" --epochs 1 --classifier "l2logreg"

:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0.001 --embed-droprate 0.1   --lr 0.001

:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001

:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.00001 --lr 0.0005
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.00005
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.0001 --lr 0.0001

:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001

:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.0001 --lr 0.0001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.0001 --lr 0.0001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.00005
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.1  --lr 0.00001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.0001

:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.00001 --lr 0.0005
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.00005
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.0001 --lr 0.0001

:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.1  --lr 0.00001
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.0001

:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.00005
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.1 --lr 0.00005
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.1 --lr 0.00001

:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.0001 --lr 0.0001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0.01 --lr 0.00005

:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.0005
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.0001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.00001
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.00005
:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.000001

:: python -m models.bert --target "dspln_PSYCHIATRY_12" --table "bert_tuning" --weight-decay 0 --lr 0.005

:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.2 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0005
:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001
:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0.001 --embed-droprate 0.1   --lr 0.001
:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001

:: python -m models.lstm --target "dspln_PSYCHIATRY_12" --table "lstm_tuning" --dropout 0.1 --wdrop 0.01  --embed-droprate 0.1   --lr 0.0001
:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0     --embed-droprate 0.001 --lr 0.0001

:: python -m models.lstm --target "dspln_SOCIALWORK_12" --table "lstm_tuning" --dropout 0.3 --wdrop 0.001 --embed-droprate 0.1   --lr 0.001
:: python -m models.bert --target "dspln_SOCIALWORK_12" --table "bert_tuning" --weight-decay 0.1 --lr 0.00005

:: call .\bats\tables\see_psych.bat
:: call .\bats\tables\see_counselling.bat