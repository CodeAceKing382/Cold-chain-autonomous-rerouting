# temperature_model.py
from __future__ import annotations
import random
from typing import Optional

from config import SimConfig
from data_models import Compartment


def update_compartment_temperature(
    compartment: Compartment,
    ambient_c: float,
    cfg: SimConfig,
    door_open: bool = False,
) -> float:
    """
    Per-compartment temperature dynamics.
    Each compartment has independent thermal control:
      - temp drifts toward ambient due to leakage (reduced by better insulation)
      - cooling pulls temp toward compartment setpoint
      - door-open creates a spike (loading/unloading)
      - occasional "glitch" reduces effective cooling
    
    Args:
        compartment: Compartment object with current temp and thermal properties
        ambient_c: Ambient temperature outside vehicle
        cfg: Simulation configuration
        door_open: Whether compartment door is open this step
    
    Returns:
        Updated temperature for this compartment
    """
    temp_c = compartment.temp_c
    
    # leakage: drift toward ambient (affected by insulation quality)
    leakage = cfg.base_leakage * compartment.insulation_quality
    leakage *= (1.0 + random.gauss(0, 0.15))
    leakage = max(0.0, leakage)

    # cooling: pull toward setpoint, scaled by efficiency
    cooling_power = cfg.base_cooling_power * compartment.cooling_efficiency
    cooling_power *= (1.0 + random.gauss(0, 0.10))
    cooling_power = max(0.0, cooling_power)

    # occasional reefer underperformance (glitch)
    if random.random() < cfg.reefer_glitch_prob:
        cooling_power *= cfg.glitch_cooling_factor

    # drift terms
    # move a fraction toward ambient and setpoint each step
    toward_ambient = leakage * (ambient_c - temp_c)
    toward_setpoint = cooling_power * (compartment.setpoint_c - temp_c)

    next_temp = temp_c + toward_ambient + toward_setpoint

    # noise
    next_temp += random.gauss(0, cfg.ambient_noise_sd)

    # door open spike
    if door_open:
        next_temp += cfg.door_open_spike_c

    return next_temp


def update_temperature_step(
    temp_c: float,
    vehicle,  # For backward compatibility
    ambient_c: float,
    cfg: SimConfig,
    door_open: bool = False,
) -> float:
    """
    DEPRECATED: Legacy single-temperature function.
    Kept for backward compatibility during transition.
    Use update_compartment_temperature for multi-compartment vehicles.
    """
    # Simple placeholder using old logic
    leakage = cfg.base_leakage * (1.0 + random.gauss(0, 0.15))
    leakage = max(0.0, leakage)

    cooling_power = cfg.base_cooling_power * 0.85
    cooling_power *= (1.0 + random.gauss(0, 0.10))
    cooling_power = max(0.0, cooling_power)

    if random.random() < cfg.reefer_glitch_prob:
        cooling_power *= cfg.glitch_cooling_factor

    toward_ambient = leakage * (ambient_c - temp_c)
    toward_setpoint = cooling_power * (6.0 - temp_c)  # Default setpoint

    next_temp = temp_c + toward_ambient + toward_setpoint
    next_temp += random.gauss(0, cfg.ambient_noise_sd)

    if door_open:
        next_temp += cfg.door_open_spike_c

    return next_temp
