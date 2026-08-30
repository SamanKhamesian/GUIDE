"""Post-policy safeguard for bolus-insulin actions."""

import numpy as np

from config import Action, BolusSafetyConfig, EnvConfig


def apply_bolus_safeguard(proposed_action, bolus_history):
    """Clip a proposed bolus before it is delivered to the simulator.

    The RL action space is unchanged: the agent still proposes Nothing, Eat,
    or Inject with a bolus in the configured proposal range. For an Inject
    action, this function calculates a dynamic upper bound from estimated
    insulin-on-board (IOB), a rolling cumulative-dose limit, and a minimum
    bolus interval. The returned action is the action executed by the
    simulator.

    IOB is estimated with linear decay over the configured active-insulin
    duration. This is an explicit simulation assumption and can later be
    replaced without changing the environment or the RL algorithms.

    Parameters
    ----------
    proposed_action : tuple
        ``(action_type, value, time_index)`` proposed by the agent.
    bolus_history : array-like
        Bolus values in real insulin units from the current simulator window,
        ordered from oldest to newest at five-minute resolution.

    Returns
    -------
    executed_action : tuple
        The safeguarded action delivered to the simulator.
    decision : dict
        Values used by the safeguard, retained for comparison and analysis.
    """
    original_action = proposed_action
    action_type, proposed_value, time_index = original_action
    action_type = int(action_type)
    proposed_value = float(proposed_value)

    # Non-insulin actions pass through the layer without modification.
    decision = {"proposed_action": original_action,
                "executed_action": original_action,
                "modified": False,
                "rejected": False,
                "reasons": [],
                "estimated_iob": 0.0,
                "rolling_cumulative_bolus": 0.0,
                "minutes_since_last_bolus": None,
                "dynamic_upper_bound": None,
                "removed_insulin": 0.0, }

    if not BolusSafetyConfig.ENABLED or action_type != Action.INJECT:
        return original_action, decision

    # Normalize only the Inject time index. Eat and Nothing actions are never
    # modified by the bolus safeguard.
    time_index = int(np.clip(round(time_index), 0, 11))

    boluses = np.asarray(bolus_history, dtype=float).reshape(-1)
    if boluses.size == 0:
        raise ValueError("bolus_history must contain at least one time point")

    # Negative or missing insulin values cannot contribute to active insulin.
    boluses = np.nan_to_num(boluses, nan=0.0, posinf=0.0, neginf=0.0)
    boluses = np.maximum(boluses, 0.0)

    # The newest history point precedes time_index=0 by one five-minute slot.
    # Adding time_index evaluates every previous bolus at the intended future
    # injection time within the upcoming hour.
    if BolusSafetyConfig.SLOT_MINUTES <= 0:
        raise ValueError("SLOT_MINUTES must be greater than zero")

    if BolusSafetyConfig.ACTIVE_INSULIN_DURATION_MINUTES <= 0:
        raise ValueError("ACTIVE_INSULIN_DURATION_MINUTES must be greater than zero")

    if BolusSafetyConfig.CUMULATIVE_WINDOW_MINUTES <= 0:
        raise ValueError("CUMULATIVE_WINDOW_MINUTES must be greater than zero")

    row_indices = np.arange(boluses.size)
    ages_minutes = (boluses.size - row_indices + time_index) * BolusSafetyConfig.SLOT_MINUTES

    # Estimate IOB using a transparent linear-decay approximation.
    active_duration = BolusSafetyConfig.ACTIVE_INSULIN_DURATION_MINUTES
    remaining_fraction = np.clip(1.0 - ages_minutes / active_duration, 0.0, 1.0)
    estimated_iob = float(np.sum(boluses * remaining_fraction))

    # Calculate bolus insulin delivered within the configured rolling window.
    in_cumulative_window = ages_minutes <= BolusSafetyConfig.CUMULATIVE_WINDOW_MINUTES
    rolling_cumulative = float(np.sum(boluses[in_cumulative_window]))

    bolus_ages = ages_minutes[boluses > 0.0]
    minutes_since_last_bolus = float(np.min(bolus_ages)) if bolus_ages.size else None

    remaining_iob_allowance = max(0.0, BolusSafetyConfig.MAX_ACTIVE_BOLUS - estimated_iob, )
    remaining_cumulative_allowance = max(0.0, BolusSafetyConfig.MAX_CUMULATIVE_BOLUS - rolling_cumulative, )

    # The stacking gate contributes an upper bound of zero when a bolus was
    # delivered too recently; otherwise it does not further restrict the dose.
    stacking_allowed = (minutes_since_last_bolus is None or minutes_since_last_bolus >= BolusSafetyConfig.MIN_BOLUS_INTERVAL_MINUTES)
    stacking_upper_bound = np.inf if stacking_allowed else 0.0

    dynamic_upper_bound = max(0.0,
        min(float(EnvConfig.INSULIN_RANGE[1]), remaining_iob_allowance, remaining_cumulative_allowance, stacking_upper_bound, ), )

    # This is the only operation that changes the proposed insulin amount.
    clipped_value = float(np.clip(proposed_value, 0.0, dynamic_upper_bound))

    if not stacking_allowed:
        decision["reasons"].append("insulin_stacking")

    if proposed_value > remaining_iob_allowance:
        decision["reasons"].append("iob_limit")

    if proposed_value > remaining_cumulative_allowance:
        decision["reasons"].append("cumulative_bolus_limit")

    # Preserve the existing executable action set: Nothing or Inject >= 2 U.
    if clipped_value < EnvConfig.INSULIN_RANGE[0]:
        executed_action = (Action.NOTHING, 0.0, time_index)
        decision["rejected"] = True

        if clipped_value > 0.0:
            decision["reasons"].append("below_minimum_executable_bolus")
    else:
        executed_action = (Action.INJECT, clipped_value, time_index)

    decision.update({"executed_action": executed_action,
                     "modified": (executed_action[0] != action_type or not np.isclose(executed_action[1], proposed_value)),
                     "estimated_iob": estimated_iob,
                     "rolling_cumulative_bolus": rolling_cumulative,
                     "minutes_since_last_bolus": minutes_since_last_bolus,
                     "dynamic_upper_bound": dynamic_upper_bound,
                     "removed_insulin": max(0.0, proposed_value - executed_action[1]), })

    return executed_action, decision
