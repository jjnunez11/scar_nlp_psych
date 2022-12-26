from subprocess import check_output, CalledProcessError

# Hyper-parameters
table = "lstm_tuning"
target = "dspln_PSYCHIATRY_60"
# Do we need bidirectional? Is it working wit bool?
num_layers = [1]
hidden_dim = [512]
dropouts = [0.1, 0.2, 0.3]
embed_droprates = [0.1, 0.01, 0.001, 0]
wdrops = [0.01, 0.001, 0.0001, 0]
lrs = [0.0001, 0.0005, 0.001]

for num_layer in num_layers:
    for dropout in dropouts:
        for embed_droprate in embed_droprates:
            for wdrop in wdrops:
                for lr in lrs:
                    command = f'python -m models.lstm --target {target} ' \
                              f'--table {table} --num-layers {num_layer}  --lr {lr}' \
                              f' --dropout {dropout} --embed-droprate {embed_droprate} --wdrop {wdrop}'
                    print(f'executing command (1): {command}')
                    try:
                        check_output(command, shell=True)
                    except CalledProcessError:
                        print(f'Error using: lr: {lr}, dropout: {dropout}, embed-droprate: {embed_droprate}'
                              f'wdrop: {wdrop}')

                    command = f'python -m models.lstm --target {target} ' \
                              f'--table {table} --num-layers {num_layer}  --lr {lr}' \
                              f' --dropout {dropout} --embed-droprate {embed_droprate} --wdrop {wdrop}'
                    print(f'executing command (2): {command}')
                    try:
                        check_output(command, shell=True)
                    except CalledProcessError:
                        print(f'Error using: lr: {lr}, dropout: {dropout}, embed-droprate: {embed_droprate}'
                              f'wdrop: {wdrop}')

