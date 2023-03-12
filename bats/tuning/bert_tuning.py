# Due to some werid bug, this would always end up errors so didn't actually use it for bert tuning

from subprocess import check_output

# Hyper-parameters
table = "bert_tuning"
target = "dspln_PSYCHIATRY_60"
weight_decays = [0, 0.0001]
lrs = [0.0001, 0.00005, 0.00001]

for lr in lrs:
    for weight_decay in weight_decays:
        command = f'python -m models.bert --target {target} --batch-size 8 ' \
                  f'--table {table}  --lr {lr} --weight-decay {weight_decay} --imbalance-fix undersampling'
        print(f'executing command: {command}')
        check_output(command, shell=True)
