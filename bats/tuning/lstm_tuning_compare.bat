:: Compare some of the top performing on grid search to determine

:: bac 72.4, AUC 79.9
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.001

:: BAC 74.3, AUC 81.4
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.01  --wdrop 0.01 --lr 0.001

:: Performance was crap
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.0001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.001  --wdrop 0 --lr 0.0001

:: BAC 73.2, AUC 81.3
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0 --lr 0.0005

:: Nooope lol.
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling

:: BAC 0.742, AUC 81.5
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --imbalance-fix undersampling --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005

:: Not using undersampling: 72.7, 79.5
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2  --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2  --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2  --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2  --embed-droprate 0.1  --wdrop 0.01 --lr 0.0005

:: Not using undersampling 70.7, 78:
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --embed-droprate 0.01  --wdrop 0.01 --lr 0.001
:: python -m models.lstm --target dspln_PSYCHIATRY_60  --dropout 0.2 --embed-droprate 0.01  --wdrop 0.01 --lr 0.001



