"""Script with Analysis functions that can be called in the run.py scripts"""

def analyse_thresholds(results):
    """Analyse threshold values from sweep results."""
    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) == 0:
        return None

    thresholds = {}

    # 1. Stability threshold (>1% failure rate)
    first_failure = next((r for r in valid_results if r['failure_rate_mean'] > 0.01), None)
    thresholds['stability'] = first_failure['shock'] if first_failure else None

    # 2. Contagion threshold (contagion rounds > 0.5)
    contagion_start = next((r for r in valid_results if r.get('contagion_rounds_mean', 0) > 0.5), None)
    thresholds['contagion'] = contagion_start['shock'] if contagion_start else None

    # 3. Critical threshold (phase transition - steepest increase)
    if len(valid_results) > 2:
        max_derivative = 0
        critical_point = None

        for i in range(1, len(valid_results)):
            derivative = abs(valid_results[i]['failure_rate_mean'] -
                           valid_results[i-1]['failure_rate_mean'])
            if derivative > max_derivative:
                max_derivative = derivative
                critical_point = valid_results[i]

        thresholds['critical'] = critical_point['shock'] if critical_point else None
    else:
        thresholds['critical'] = None

    # 4. Systemic crisis threshold (>30% failures)
    systemic_crisis = next((r for r in valid_results if r['failure_rate_mean'] > 0.30), None)
    thresholds['systemic_crisis'] = systemic_crisis['shock'] if systemic_crisis else None

    # 5. Collapse threshold (>50% failures)
    collapse = next((r for r in valid_results if r['failure_rate_mean'] > 0.50), None)
    thresholds['collapse'] = collapse['shock'] if collapse else None

    return thresholds