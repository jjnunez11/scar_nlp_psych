from subprocess import check_output

# Hyper-parameters
table = "bow_tuning"
# target = "dspln_PSYCHIATRY_12"
target = "dspln_SOCIALWORK_12"
max_tokens = [5000]  # 1000
# c_s = [0.1, 0.2, 0.3, 0.5]
# c_s = [0.6, 0.7, 0.8, 0.9, 1]
# c_s = [1, 3, 5, 10, 100]
# c_s = [0.1, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 3, 5, 10]
# c_s = [0.75, 0.85, 0.95, 1.05, 1.15]
# c_s = [0.1, 0.5, 0.8, 1, 1.02, 1.04, 1.06, 1.08, 1.12, 1.14, 1.25, 1.5, 2, 5]
# c_s = [0.3, 0.4, 0.5, 0.6]
# c_s = [0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]
c_s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 5]
estimators = [50, 100, 200]

for max_token in max_tokens:
    classifier = "l2logreg"
    for c in c_s:
        command = f'python -m models.bow --target {target} --max-tokens {max_token} '\
                  f'--table {table}  --classifier {classifier} --l2logreg-c {c} --epochs 1'
        print(f'executing command: {command}')
        check_output(command, shell=True)

    # classifier = "rf"
    # for estimator in estimators:
    #    command = f'python -m models.bow --target {target} --max-tokens {max_token} ' \
    #              f'--table {table}  --classifier {classifier} --rf-estimators {estimator} --epochs 1'
    #    print(f'executing command: {command}')
    #    check_output(command, shell=True)

