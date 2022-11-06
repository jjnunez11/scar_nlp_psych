from subprocess import check_output, CalledProcessError

# Hyper-parameters
table = "lstm_tuning"
target = "dspln_PSYCHIATRY_60"
# Do we need bidirectional? Is it working wit bool?
num_layers = [1]
hidden_dim = [512]
dropouts = [0.2]  # 0.1, 0.2, 0.5, 0.8 - got an error when doing 0.5, so may want to retry that
embed_droprates = [0.1, 0.01, 0.001, 0]  # 0.2
wdrops = [0.01, 0.001, 0.0001, 0]
lrs = [0.0001, 0.0005, 0.001]

for num_layer in num_layers:
    for dropout in dropouts:
        for embed_droprate in embed_droprates:
            for wdrop in wdrops:
                for lr in lrs:
                    command = f'python -m models.lstm --target {target} ' \
                              f'--table {table} --num-layers {num_layer}  --lr {lr}' \
                              f' --dropout {dropout} --embed-droprate {embed_droprate} --wdrop {wdrop}' \
                              f' --imbalance-fix undersampling'
                    print(f'executing command: {command}')
                    try:
                        check_output(command, shell=True)
                    except CalledProcessError:
                        print(f'Error using: lr: {lr}, dropout: {dropout}, embed-droprate: {embed_droprate}'
                              f'wdrop: {wdrop}')

