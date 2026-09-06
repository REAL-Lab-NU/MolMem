"""Molecular property descriptions shared by SFT and RL prompts."""

PROPERTY_FULL_NAMES = {
    "qed": ["QED", "Quantitative Estimate of Drug-likeness (QED)", "drug-likeness quantified by QED score"],
    "logp": ["LogP", "Octanol-water partition coefficient (logP)", "lipophilicity measured by logP"],
    "jnk3": ["JNK3 inhibition", "c-Jun N-terminal kinase 3 inhibition probability", "inhibition probability of c-Jun N-terminal kinase 3"],
    "drd2": ["DRD2 inhibition", "Dopamine receptor D2 inhibition probability", "inhibition probability of Dopamine receptor D2"],
    "sa": ["SA score", "Synthetic accessibility score", "synthetic accessibility score (lower is better)"]
}

PROPERTY_OPTIMIZATION_DIRECTIONS = {
    "qed": "maximize",
    "logp": "maximize",
    "jnk3": "maximize",
    "drd2": "maximize",
    "sa": "minimize"
}

def generate_property_description(task: str, instr_setting: str = 'seen') -> str:
    """
    Build the property objective used by the SFT prompt.

    Args:
        task: Task string such as ``qed+logp+sa``.
        instr_setting: ``seen`` or ``unseen`` property naming style.

    Returns:
        A concise property objective.
    """
    properties = task.split('+')
    descriptions = []

    for prop in properties:
        if prop not in PROPERTY_FULL_NAMES:
            raise ValueError(f"Unknown property: {prop}. Supported properties: {list(PROPERTY_FULL_NAMES.keys())}")

        if instr_setting == 'unseen':
            prop_name = PROPERTY_FULL_NAMES[prop][-1]
        else:
            prop_name = PROPERTY_FULL_NAMES[prop][0]

        change_direction = 'decrease' if prop == 'sa' else 'increase'
        descriptions.append(f"{change_direction} {prop_name}")

    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1]}"
    else:
        return ', '.join(descriptions[:-1]) + f" and {descriptions[-1]}"

def parse_task_config(env_config: dict) -> tuple[str, str]:
    """
    Parse the task description and similarity threshold.

    Args:
        env_config: Environment configuration dictionary.

    Returns:
        tuple: (task_description, similarity_threshold_str)
    """
    task = env_config.get('molecule_opt_task')
    if not isinstance(task, str) or not task.strip():
        raise ValueError("molecule_opt_task must be configured")
    task = task.strip()
    similarity_threshold = env_config.get('molecule_opt_similarity_threshold', 0.4)

    task_description = generate_property_description(task, 'seen')

    similarity_threshold_str = str(similarity_threshold)

    return task_description, similarity_threshold_str

def get_supported_properties() -> list[str]:
    """Return all supported properties."""
    return list(PROPERTY_FULL_NAMES.keys())

def get_optimization_directions(task: str) -> dict[str, str]:
    """
    Return the optimization direction of every task property.

    Args:
        task: Task string such as ``qed+logp+sa``.

    Returns:
        A property-to-direction mapping.
    """
    properties = task.split('+')
    directions = {}
    for prop in properties:
        if prop in PROPERTY_OPTIMIZATION_DIRECTIONS:
            directions[prop] = PROPERTY_OPTIMIZATION_DIRECTIONS[prop]
        else:
            directions[prop] = "maximize"
    return directions

def calculate_success_rates(task: str, best_results_dict: dict, improvement_based: bool = True) -> dict:
    """
    Calculate success rates using improvement or absolute thresholds.

    Args:
        task: Single- or multi-property task string.
        best_results_dict: Best result for each source molecule.
        improvement_based: Use improvement thresholds when true.

    Returns:
        Success-rate metrics.
    """
    if not best_results_dict:
        return {}

    properties = task.split('+')
    results = {}
    total_molecules = len(best_results_dict)

    improvement_success_counts = {prop: 0 for prop in properties}
    absolute_success_counts = {prop: 0 for prop in properties}
    improvement_overall_success = 0
    absolute_overall_success = 0

    for smiles, data in best_results_dict.items():
        if 'best_score' not in data or 'initial_score' not in data:
            continue

        improvement_all_success = True
        absolute_all_success = True

        if len(properties) > 1:
            individual_improvements = data.get('individual_improvements', {})
            individual_new_properties = data.get('individual_new_properties', {})

            for prop in properties:
                if individual_improvements:
                    improvement = individual_improvements.get(prop, 0)
                    if isinstance(improvement, dict):
                        improvement = improvement.get('improvement', 0)

                    if is_property_success(prop, improvement):
                        improvement_success_counts[prop] += 1
                    else:
                        improvement_all_success = False
                else:
                    improvement_all_success = False

                if individual_new_properties:
                    absolute_value = individual_new_properties.get(prop)
                    if absolute_value is not None:
                        if is_property_absolute_success(prop, absolute_value):
                            absolute_success_counts[prop] += 1
                        else:
                            absolute_all_success = False
                    else:
                        absolute_all_success = False
                else:
                    absolute_all_success = False

        else:
            prop = properties[0]
            best_score = data['best_score']
            initial_score = data['initial_score']
            improvement = best_score - initial_score

            if is_property_success(prop, improvement):
                improvement_success_counts[prop] += 1
            else:
                improvement_all_success = False

            if is_property_absolute_success(prop, best_score):
                absolute_success_counts[prop] += 1
            else:
                absolute_all_success = False

        if improvement_all_success:
            improvement_overall_success += 1

        if absolute_all_success:
            absolute_overall_success += 1

    if improvement_based:
        for prop in properties:
            results[f"{prop}_success_rate"] = improvement_success_counts[prop] / total_molecules

        if len(properties) > 1:
            results["overall_success_rate"] = improvement_overall_success / total_molecules
    else:
        for prop in properties:
            results[f"{prop}_absolute_success_rate"] = absolute_success_counts[prop] / total_molecules

        if len(properties) > 1:
            results["overall_absolute_success_rate"] = absolute_overall_success / total_molecules

    return results

def calculate_both_success_rates(task: str, best_results_dict: dict) -> dict:
    """
    Calculate both improvement-based and absolute success rates.

    Args:
        task: Task string.
        best_results_dict: Best result for each source molecule.

    Returns:
        Combined success-rate metrics.
    """
    if not best_results_dict:
        return {}

    improvement_results = calculate_success_rates(task, best_results_dict, improvement_based=True)

    absolute_results = calculate_success_rates(task, best_results_dict, improvement_based=False)

    results = {}
    results.update(improvement_results)
    results.update(absolute_results)

    return results

def calculate_property_averages(task: str, best_results_dict: dict) -> dict:
    """
    Calculate mean improvement and best score for each property.

    Args:
        task: Single- or multi-property task string.
        best_results_dict: Best result for each source molecule.

    Returns:
        Per-property average metrics.
    """
    if not best_results_dict:
        return {}

    properties = task.split('+')
    results = {}

    if len(properties) == 1:
        prop = properties[0]
        improvements = []
        best_scores = []

        for data in best_results_dict.values():
            if 'best_score' in data and 'initial_score' in data:
                improvement = data['best_score'] - data['initial_score']
                improvements.append(improvement)
                best_scores.append(data['best_score'])

        if improvements:
            results[f"{prop}_avg_improvement"] = sum(improvements) / len(improvements)
            results[f"{prop}_avg_best_score"] = sum(best_scores) / len(best_scores)

    else:
        property_improvements = {prop: [] for prop in properties}
        property_best_scores = {prop: [] for prop in properties}

        for data in best_results_dict.values():
            individual_improvements = data.get('individual_improvements', {})
            individual_old_properties = data.get('individual_old_properties', {})
            individual_new_properties = data.get('individual_new_properties', {})

            for prop in properties:
                if prop in individual_new_properties:
                    property_best_scores[prop].append(individual_new_properties[prop])

                    if prop in individual_old_properties:
                        improvement = individual_new_properties[prop] - individual_old_properties[prop]
                        property_improvements[prop].append(improvement)
                    elif prop in individual_improvements:
                        property_improvements[prop].append(individual_improvements[prop])

                elif prop in individual_improvements and prop in individual_old_properties:
                    improvement = individual_improvements[prop]
                    property_improvements[prop].append(improvement)

                    best_value = individual_old_properties[prop] + improvement
                    property_best_scores[prop].append(best_value)

                elif prop in individual_improvements:
                    prop_data = individual_improvements[prop]
                    if isinstance(prop_data, dict):
                        if 'improvement' in prop_data:
                            property_improvements[prop].append(prop_data['improvement'])
                        if 'best' in prop_data:
                            property_best_scores[prop].append(prop_data['best'])
                    else:
                        property_improvements[prop].append(prop_data)

        for prop in properties:
            if property_improvements[prop]:
                avg_improvement = sum(property_improvements[prop]) / len(property_improvements[prop])
                results[f"{prop}_avg_improvement"] = avg_improvement

            if property_best_scores[prop]:
                avg_best_score = sum(property_best_scores[prop]) / len(property_best_scores[prop])
                results[f"{prop}_avg_best_score"] = avg_best_score

    return results

def is_property_success(property_name: str, improvement: float) -> bool:
    """
    Test whether an improvement reaches its property threshold.

    Args:
        property_name: Property name.
        improvement: ``best_score - initial_score``.

    Returns:
        Whether the threshold is reached.
    """
    config = PROPERTY_SUCCESS_THRESHOLDS.get(property_name)
    if not config:
        return False

    threshold = config["threshold"]
    direction = config["direction"]

    if direction == "increase":
        return improvement >= threshold
    else:
        return improvement <= -threshold

def is_property_absolute_success(property_name: str, absolute_value: float) -> bool:
    """
    Test whether an absolute property value reaches its threshold.

    Args:
        property_name: Property name.
        absolute_value: Absolute property value.

    Returns:
        Whether the threshold is reached.
    """
    config = PROPERTY_ABSOLUTE_THRESHOLDS.get(property_name)
    if not config:
        return False

    threshold = config["threshold"]
    direction = config["direction"]

    if direction == "increase":
        return absolute_value >= threshold
    else:
        return absolute_value <= threshold

PROPERTY_DIFFICULTY_WEIGHTS = {
    "jnk3": 15,
    "qed": 20,
    "drd2": 4,
    "sa": 2,
    "logp": 1
}

PROPERTY_SUCCESS_THRESHOLDS = {
    "qed": {"threshold": 0.1, "direction": "increase"},
    "logp": {"threshold": 1.0, "direction": "increase"},
    "jnk3": {"threshold": 0.1, "direction": "increase"},
    "drd2": {"threshold": 0.5, "direction": "increase"},
    "sa": {"threshold": 0.5, "direction": "decrease"}
}

PROPERTY_ABSOLUTE_THRESHOLDS = {
    "qed": {"threshold": 0.9, "direction": "increase"},
    "logp": {"threshold": 2.0, "direction": "increase"},
    "drd2": {"threshold": 0.8, "direction": "increase"},
    "jnk3": {"threshold": 0.1, "direction": "increase"},
    "sa": {"threshold": 2.5, "direction": "decrease"}
}

def calculate_multi_objective_reward(improvements: dict, task_properties: list, optimization_directions: dict = None) -> tuple[float, dict]:
    """
    Calculate the multi-objective reward.

    Args:
        improvements: Improvement value for each property.
        task_properties: Properties in the task.
        optimization_directions: Property-to-direction mapping.

    Returns:
        tuple: (total_reward, reward_info)
    """
    if optimization_directions is None:
        optimization_directions = {prop: PROPERTY_OPTIMIZATION_DIRECTIONS.get(prop, "maximize")
                                 for prop in task_properties}

    base_reward = 0.0
    weighted_improvements = {}

    for prop in task_properties:
        improvement = improvements.get(prop, 0.0)
        weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
        direction = optimization_directions.get(prop, "maximize")

        if direction == "minimize":
            weighted_improvement = (-improvement) * weight
        else:
            weighted_improvement = improvement * weight

        weighted_improvements[prop] = weighted_improvement
        base_reward += weighted_improvement

    success_bonus = 0.0
    success_status = {}
    all_success = True

    for prop in task_properties:
        improvement = improvements.get(prop, 0.0)
        config = PROPERTY_SUCCESS_THRESHOLDS.get(prop)

        if config:
            threshold = config["threshold"]
            direction = config["direction"]

            if direction == "increase":
                success = improvement >= threshold
            else:
                success = improvement <= -threshold

            success_status[prop] = {
                "success": success,
                "improvement": improvement,
                "threshold": threshold,
                "direction": direction
            }

            if not success:
                all_success = False
        else:
            success_status[prop] = {"success": True, "improvement": improvement}

    if all_success and len(task_properties) > 1:
        success_bonus = 5.0 + len(task_properties) * 2.0

    total_reward = base_reward + success_bonus

    reward_info = {
        "base_reward": base_reward,
        "success_bonus": success_bonus,
        "total_reward": total_reward,
        "weighted_improvements": weighted_improvements,
        "success_status": success_status,
        "all_success": all_success
    }

    return total_reward, reward_info

PROPERTY_IMPROVEMENT_GUIDANCE = {
    "qed": {
        "positive": "Great QED improvement! The molecule is becoming more drug-like.",
        "negative": "QED decreased. Try to improve QED.",
        "maintain": "Excellent drug-likeness maintained."
    },
    "logp": {
        "positive": "Good LogP increase! Enhanced lipophilicity will improve membrane permeability.",
        "negative": "LogP decreased. Try to improve LogP.",
        "maintain": "Good lipophilicity balance maintained."
    },
    "sa": {
        "positive": "Excellent! SA score decreased, making the molecule easier to synthesize.",
        "negative": "SA score increased, making synthesis more challenging. Try to decrease SA score.",
        "maintain": "Good synthetic accessibility maintained."
    },
    "jnk3": {
        "positive": "Great JNK3 activity improvement! Enhanced kinase inhibition potential.",
        "negative": "JNK3 activity decreased. Try to improve JNK3 inhibition.",
        "maintain": "Strong JNK3 activity maintained."
    },
    "drd2": {
        "positive": "Great DRD2 activity improvement! Better dopamine receptor interaction.",
        "negative": "DRD2 activity decreased. Try to improve DRD2 inhibition.",
        "maintain": "Strong DRD2 activity maintained."
    }
}

def generate_multi_objective_guidance(reward_info: dict) -> str:
    """
    Generate guidance for a multi-objective task.

    Args:
        reward_info: Reward details and success status.

    Returns:
        Guidance text.
    """
    if not reward_info.get("is_multi_objective", False):
        return ""

    improvements = reward_info.get("improvements", {})
    success_status = reward_info.get("success_status", {})
    all_success = reward_info.get("all_success", False)
    success_bonus = reward_info.get("success_bonus", 0)

    positive_changes = []
    negative_changes = []

    for prop, improvement in improvements.items():
        status = success_status.get(prop, {})
        is_success = status.get("success", False)

        if improvement > 0:
            if prop == 'sa':
                negative_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": PROPERTY_IMPROVEMENT_GUIDANCE[prop]["negative"]
                })
            else:
                message = PROPERTY_IMPROVEMENT_GUIDANCE[prop]["positive"]
                if is_success:
                    message += f" 🎯 Threshold achieved (+{improvement:.3f} ≥ {status.get('threshold', 0)})!"

                positive_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": message,
                    "threshold_achieved": is_success
                })

        elif improvement < 0:
            if prop == 'sa':
                message = PROPERTY_IMPROVEMENT_GUIDANCE[prop]["positive"]
                if is_success:
                    message += f" 🎯 Threshold achieved ({improvement:.3f} ≤ -{status.get('threshold', 0)})!"

                positive_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": message,
                    "threshold_achieved": is_success
                })
            else:
                negative_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": PROPERTY_IMPROVEMENT_GUIDANCE[prop]["negative"]
                })

    guidance_parts = []

    if all_success and success_bonus > 0:
        guidance_parts.append(f"🎉 OUTSTANDING! All targets achieved! Success bonus: +{success_bonus:.1f}")

    if positive_changes:
        if len(positive_changes) == 1:
            guidance_parts.append(positive_changes[0]["message"])
        else:
            messages = [change["message"] for change in positive_changes]
            guidance_parts.append(" ".join(messages))

    if negative_changes:
        if positive_changes:
            guidance_parts.append("However, some properties need attention:")
        for change in negative_changes:
            guidance_parts.append(change["message"])

    if all_success:
        guidance_parts.append("Continue with similar modification strategies!")
    elif positive_changes and not negative_changes:
        guidance_parts.append("Excellent progress! Keep refining with this approach.")
    elif positive_changes and negative_changes:
        guidance_parts.append("Focus on maintaining the positive changes while addressing the declining properties.")
    else:
        guidance_parts.append("Consider trying different structural modifications.")

    return " ".join(guidance_parts)

def format_property_scores(original_scores: dict, new_scores: dict, task_properties: list) -> str:
    """
    Format source and candidate property scores.

    Args:
        original_scores: Source property scores.
        new_scores: Candidate property scores.
        task_properties: Properties in the task.

    Returns:
        Formatted score text.
    """
    score_parts = []

    for prop in task_properties:
        original_score = original_scores.get(prop, 0.0)
        new_score = new_scores.get(prop, 0.0)
        improvement = new_score - original_score

        if improvement > 0:
            if prop == 'sa':
                change_symbol = "↑"
            else:
                change_symbol = "↑"
        elif improvement < 0:
            if prop == 'sa':
                change_symbol = "↓"
            else:
                change_symbol = "↓"
        else:
            change_symbol = "→"

        prop_name = PROPERTY_FULL_NAMES[prop][0]

        score_parts.append(f"{prop_name}: {original_score:.3f} → {new_score:.3f} ({change_symbol}{abs(improvement):.3f})")

    return " | ".join(score_parts)

def generate_complete_multi_objective_guidance(
    original_scores: dict,
    new_scores: dict,
    task_properties: list,
    reward_info: dict,
    is_initial: bool = False
) -> str:
    """
    Generate property scores and guidance for a multi-objective task.

    Args:
        original_scores: Source property scores.
        new_scores: Candidate property scores.
        task_properties: Properties in the task.
        reward_info: Reward details.
        is_initial: Whether this is the initial source display.

    Returns:
        Complete guidance text.
    """
    guidance_parts = []

    if is_initial:
        original_score_str = ", ".join([
            f"{PROPERTY_FULL_NAMES[prop][0]}: {new_scores.get(prop, 0.0):.3f}"
            for prop in task_properties
        ])
        guidance_parts.append(f"Original molecule properties: {original_score_str}")
        return " ".join(guidance_parts)

    if 'global_improvements' in reward_info and 'original_properties' in reward_info:
        global_improvements = reward_info['global_improvements']
        original_properties = reward_info['original_properties']

        global_parts = []
        for prop in task_properties:
            if prop in global_improvements and prop in original_properties:
                global_imp = global_improvements[prop]
                orig_val = original_properties[prop]
                current_val = new_scores.get(prop, 0.0)

                if prop == 'sa':
                    symbol = "↓" if global_imp < 0 else "↑"
                    is_good = global_imp < 0
                else:
                    symbol = "↑" if global_imp > 0 else "↓"
                    is_good = global_imp > 0

                color_indicator = "✓" if is_good else "✗"
                global_parts.append(f"{PROPERTY_FULL_NAMES[prop][0]}: {orig_val:.3f}→{current_val:.3f} ({symbol}{abs(global_imp):.3f}{color_indicator})")

        if global_parts:
            guidance_parts.append(f"Properties (vs original): {' | '.join(global_parts)}")
    else:
        score_display = format_property_scores(original_scores, new_scores, task_properties)
        guidance_parts.append(f"Property changes: {score_display}")

    if len(task_properties) > 1:
        reward_info['is_multi_objective'] = True
        if 'global_improvements' in reward_info:
            improvements = reward_info['global_improvements']
        else:
            improvements = {prop: new_scores.get(prop, 0.0) - original_scores.get(prop, 0.0)
                           for prop in task_properties}
        reward_info['improvements'] = improvements

        if 'weighted_score' in reward_info:
            guidance_parts.append(f"Weighted score: {reward_info['weighted_score']:.3f}")

        guidance_advice = generate_multi_objective_guidance(reward_info)
        if guidance_advice:
            guidance_parts.append(guidance_advice)

    return " ".join(guidance_parts)

def validate_task(task: str) -> bool:
    """Return whether a task string is valid."""
    try:
        properties = task.split('+')
        return all(prop in PROPERTY_FULL_NAMES for prop in properties)
    except:
        return False
