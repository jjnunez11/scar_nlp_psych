:: python -m models.bow --target "dspln_PSYCHIATRY_60_male" --table "compare_sexes_own_model" --epochs 1 --table_extra "Male"
:: python -m models.cnn --target "dspln_PSYCHIATRY_60_male" --table "compare_sexes_own_model" --table_extra "Male"
:: python -m models.bow --target "dspln_PSYCHIATRY_60_female" --table "compare_sexes_own_model" --epochs 1 --table_extra "Female"
:: python -m models.cnn --target "dspln_PSYCHIATRY_60_female" --table "compare_sexes_own_model" --table_extra "Female"

:: python -m models.bow --target "dspln_SOCIALWORK_60_male" --table "compare_sexes_own_model" --epochs 1 --table_extra "Male"
:: python -m models.cnn --target "dspln_SOCIALWORK_60_male" --table "compare_sexes_own_model" --table_extra "Male"
:: python -m models.bow --target "dspln_SOCIALWORK_60_female" --table "compare_sexes_own_model" --epochs 1 --table_extra "Female"
:: python -m models.cnn --target "dspln_SOCIALWORK_60_female" --table "compare_sexes_own_model" --table_extra "Female"

:: Repeat, but using a model trained on all data instead of sex separated
python -m models.bow --target "dspln_PSYCHIATRY_60_male" --table "compare_sexes_general_model" --epochs 1 --table_extra "Male"
python -m models.cnn --target "dspln_PSYCHIATRY_60_male" --table "compare_sexes_general_model" --table_extra "Male"
python -m models.bow --target "dspln_PSYCHIATRY_60_female" --table "compare_sexes_general_model" --epochs 1 --table_extra "Female"
python -m models.cnn --target "dspln_PSYCHIATRY_60_female" --table "compare_sexes_general_model" --table_extra "Female"

python -m models.bow --target "dspln_SOCIALWORK_60_male" --table "compare_sexes_general_model" --epochs 1 --table_extra "Male"
python -m models.cnn --target "dspln_SOCIALWORK_60_male" --table "compare_sexes_general_model" --table_extra "Male"
python -m models.bow --target "dspln_SOCIALWORK_60_female" --table "compare_sexes_general_model" --epochs 1 --table_extra "Female"
python -m models.cnn --target "dspln_SOCIALWORK_60_female" --table "compare_sexes_general_model" --table_extra "Female"


:: DONE