from subprocess import check_output

# Hyper-parameters
table = "cnn_tuning"  # _post_spec_fix"
target = "dspln_SOCIALWORK_12"
channels = [500]
# dropouts = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# dropouts = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
dropouts = [0.85]
# dropouts = [0.4, 0.5, 0.6, 0.7, 0.8]  # , 0.9] [0.55, 0.65]
# dropouts = [0.8]
weight_decays = [0.0001]
# weight_decays = [0]  # [0.0001, 0.001, 0]  # [0.0001]
# lrs = [0.0001, 0.00005]  # [0.0001]
# lrs = [0.00005]
lrs = [0.0001]

reps = 2

# best so far: 0.8 dropout with 0.0001 lr and weight decay 0.0001

# Check the other survival lengths, e.g. with 0.00001, 0.00005, 0.0001 and 0.2, 0.5, 0.8 dropout.

for channel in channels:
    for dropout in dropouts:
        for weight_decay in weight_decays:
            for lr in lrs:
                for i in range(reps):
                    command = f'python -m models.cnn --target {target} ' \
                              f'--table {table} --output-channel {channel} --weight-decay {weight_decay} --lr {lr}' \
                              f' --dropout {dropout}'
                    print(f'executing command: {command}')
                    check_output(command, shell=True)
