from subprocess import check_output

# Hyper-parameters
table = "cnn_tuning_post_spec_fix"
target = "dspln_PSYCHIATRY_60"
channels = [500]
# dropouts = [0.2, 0.5, 0.8]
dropouts = [0.6, 0.7, 0.8]  # , 0.9]
# dropouts = [0.8]
weight_decays = [0.0001, 0]
lrs = [0.00005, 0.0001]
# lrs = [0.00005]

# best so far: 0.8 dropout with 0.0001 lr and weight decay 0.0001

# Check the other survival lengths, e.g. with 0.00001, 0.00005, 0.0001 and 0.2, 0.5, 0.8 dropout.
for channel in channels:
    for dropout in dropouts:
        for weight_decay in weight_decays:
            for lr in lrs:
                command = f'python -m models.cnn --target {target} ' \
                          f'--table {table} --output-channel {channel} --weight-decay {weight_decay} --lr {lr}' \
                          f' --dropout {dropout}'
                print(f'executing command: {command}')
                check_output(command, shell=True)
