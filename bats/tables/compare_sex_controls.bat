:: Repeat, but using a model trained on all data instead of sex separated
python -m models.cnn --target "dspln_PSYCHIATRY_60_male_controls" --table "compare_sex_controls" --table_extra "Male control train and dev"
python -m models.cnn --target "dspln_PSYCHIATRY_60_female_controls" --table "compare_sex_controls" --table_extra "Female control train and dev"
python -m models.cnn --target "dspln_PSYCHIATRY_60_male_controls_dev" --table "compare_sex_controls" --table_extra "Both train and control dev"
python -m models.cnn --target "dspln_PSYCHIATRY_60_female_controls_dev" --table "compare_sex_controls" --table_extra "Both train and control dev"


:: DONE