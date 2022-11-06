:: tbl_see_psych
:: Performance predicting whether a patient will see a psychiatrist in first 5 years (60 months)
python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1
python -m models.cnn --target "dspln_PSYCHIATRY_60" --table "see_psych"
python -m models.lstm --target "dspln_PSYCHIATRY_60" --table "see_psych"
python -m models.bert --target "dspln_PSYCHIATRY_60" --table "see_psych"
