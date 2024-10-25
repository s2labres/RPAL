
def extract_recovery_data(experiment, baseline, deviation):
    """ Identifies the month in testing where results surpass/converge on baseline.
    Args:
        experiment: The poisoned performance that is recovering
        baseline: The baseline performance to judge the recovery against
        deviation: The acceptable +/- deviation

    Returns:
        (int, int):
            int: month of convergence initial intercept
            int: recovery rate

    """

    exp_f1 = experiment["f1"]
    base_f1 = baseline["f1"]

    initial_intercept = -1
    recovery_months = 0

    for i in range(0, len(exp_f1)):
        if initial_intercept == -1 and exp_f1[i] >= (base_f1[i] - deviation):
            initial_intercept = i
        if initial_intercept != -1 and exp_f1[i] >= (base_f1[i] - deviation):
            recovery_months += 1

    recovery_rate = recovery_months / (len(exp_f1) - initial_intercept)

    if initial_intercept > -1:
        initial_intercept += 1

    return initial_intercept, recovery_rate
