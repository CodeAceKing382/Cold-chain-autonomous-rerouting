# monitoring.py
from __future__ import annotations
from dataclasses import replace

from data_models import ExposureMetrics, ProduceBatch


def update_exposure_metrics(
    metrics: ExposureMetrics,
    temp_c: float,
    batch: ProduceBatch,
    dt_min: float,
) -> ExposureMetrics:
    """
    Updates monitoring metrics for one time step.

    Requirements satisfied:
      - threshold violations
      - duration above threshold (not just instant alerts)
      - cumulative temperature abuse (area above threshold)
      - max continuous excursion tracking
    """
    safe_min = batch.safe_min_c
    safe_max = batch.safe_max_c
    critical = batch.critical_c

    above_safe = temp_c > safe_max
    above_critical = temp_c > critical

    above_safe_minutes = metrics.above_safe_minutes
    above_critical_minutes = metrics.above_critical_minutes
    cumulative_abuse = metrics.cumulative_abuse
    current_exc = metrics.current_excursion_min
    max_exc = metrics.max_continuous_excursion_min

    if above_safe:
        above_safe_minutes += dt_min
        current_exc += dt_min

        # abuse area: how far above safe max, times dt
        cumulative_abuse += (temp_c - safe_max) * dt_min
    else:
        # excursion ends
        if current_exc > max_exc:
            max_exc = current_exc
        current_exc = 0.0

    if above_critical:
        above_critical_minutes += dt_min

    # update max excursion if still in excursion
    if current_exc > max_exc:
        max_exc = current_exc

    return ExposureMetrics(
        above_safe_minutes=above_safe_minutes,
        above_critical_minutes=above_critical_minutes,
        max_continuous_excursion_min=max_exc,
        current_excursion_min=current_exc,
        cumulative_abuse=cumulative_abuse,
    )


def estimate_quality_remaining(
    batch: ProduceBatch,
    metrics: ExposureMetrics,
) -> float:
    """
    Optional but useful: quality remaining proxy in [0,1].
    Simple mapping:
      - use cumulative_abuse scaled by k_abuse and shelf life.

    TODO:
      - replace with validated decay model (Arrhenius / empirical)
    """
    # convert shelf-life hours to "abuse budget" units (very rough)
    abuse_budget = batch.shelf_life_h * 60.0  # minutes baseline
    abuse_budget *= (1.0 / max(0.1, batch.k_abuse))  # sensitive produce: smaller budget

    # normalize cumulative abuse (units: degree-min)
    # This is a proxy; we keep it monotonic.
    frac_lost = min(1.0, metrics.cumulative_abuse / max(1.0, abuse_budget))
    return max(0.0, 1.0 - frac_lost)
