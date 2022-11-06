:: python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0 --imbalance-fix undersampling
:: python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.01 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.0001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.00001 --imbalance-fix undersampling

python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00001 --weight-decay 0.01 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00001 --weight-decay 0.001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00001 --weight-decay 0.0001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.00001 --weight-decay 0.00001 --imbalance-fix undersampling

python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.0001 --weight-decay 0.01 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.0001 --weight-decay 0.001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.0001 --weight-decay 0.0001 --imbalance-fix undersampling
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "bert_tuning"  --lr 0.0001 --weight-decay 0.00001 --imbalance-fix undersampling


:: python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0 --imbalance-fix undersampling --batch-size 4