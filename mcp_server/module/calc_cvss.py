def calculate_cvss(metrics):
    """
    Calculate the CVSS base score based on the provided metrics.

    :param metrics: A dictionary containing the CVSS metrics with keys:
                    'AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A'
    :return: The CVSS base score as a float.
    """
    exploitability_coefficient = 8.22
    scope_coefficient = 1.08

    # Define associative arrays mapping each metric value to the constant used in the CVSS scoring formula.
    weight = {
        'AV': {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.2},
        'AC': {'H': 0.44, 'L': 0.77},
        'PR': {
            'U': {'N': 0.85, 'L': 0.62, 'H': 0.27},  # Scope Unchanged
            'C': {'N': 0.85, 'L': 0.68, 'H': 0.5}    # Scope Changed
        },
        'UI': {'N': 0.85, 'R': 0.62},
        'S': {'U': 6.42, 'C': 7.52},
        'C': {'N': 0, 'L': 0.22, 'H': 0.56},
        'I': {'N': 0, 'L': 0.22, 'H': 0.56},
        'A': {'N': 0, 'L': 0.22, 'H': 0.56}
    }

    def round_up(input_value):
        """Round up to the nearest 0.1 as per the CVSS specification."""
        int_input = int(input_value * 100000)
        if int_input % 10000 == 0:
            return int_input / 100000
        else:
            return (int(int_input / 10000) + 1) / 10

    try:
        # Extract metric weights
        metric_weight = {key: weight[key][metrics[key]] for key in ['AV', 'AC', 'UI', 'C', 'I', 'A']}
        metric_weight['PR'] = weight['PR'][metrics['S']][metrics['PR']]
        metric_weight['S'] = weight['S'][metrics['S']]

        # Calculate impact sub-score
        impact_sub_score_multiplier = (1 - ((1 - metric_weight['C']) * (1 - metric_weight['I']) * (1 - metric_weight['A'])))
        if metrics['S'] == 'U':
            impact_sub_score = metric_weight['S'] * impact_sub_score_multiplier
        else:
            impact_sub_score = metric_weight['S'] * (impact_sub_score_multiplier - 0.029) - \
                               3.25 * (impact_sub_score_multiplier - 0.02) ** 15

        # Calculate exploitability sub-score
        exploitability_sub_score = exploitability_coefficient * metric_weight['AV'] * metric_weight['AC'] * \
                                   metric_weight['PR'] * metric_weight['UI']

        # Calculate base score
        if impact_sub_score <= 0:
            base_score = 0
        else:
            if metrics['S'] == 'U':
                base_score = min((exploitability_sub_score + impact_sub_score), 10)
            else:
                base_score = min((exploitability_sub_score + impact_sub_score) * scope_coefficient, 10)

        # Round up to one decimal place
        return round_up(base_score)

    except KeyError as e:
        raise ValueError(f"Invalid metric value: {e}")
