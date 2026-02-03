# data_models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal


NodeType = Literal["hub", "city_point"]


@dataclass
class Node:
    node_id: int
    node_type: NodeType
    name: str
    x: float
    y: float
    tw_open: float
    tw_close: float
    service_min: float


@dataclass
class Compartment:
    compartment_id: str          # "A", "B", "C"
    setpoint_c: float            # Target temperature for this compartment
    capacity: int                # Max units this compartment can hold
    temp_c: float = 10.0         # Current temperature (dynamic)
    cooling_efficiency: float = 0.85  # 0..1
    insulation_quality: float = 1.0   # 1.0 = standard, <1.0 = better insulation


@dataclass
class Vehicle:
    vehicle_id: int
    capacity: int                # Total vehicle capacity (sum of compartments)
    max_speed_factor: float      # 0..1+ (affects travel progress in simulation)
    # Multi-compartment structure
    compartments: Dict[str, Compartment] = field(default_factory=dict)


@dataclass
class ProduceBatch:
    batch_id: int
    produce_type: str
    safe_min_c: float
    safe_max_c: float
    critical_c: float
    shelf_life_h: float
    k_abuse: float
    priority: int              # 1=high, 2=med, 3=low


@dataclass
class Shipment:
    shipment_id: int
    customer_node_id: int
    demand_units: int
    batch: ProduceBatch
    assigned_compartment: Optional[str] = None  # Which compartment carries this shipment


@dataclass
class PlanningInstance:
    # Nodes for VRPTW: start hub (depot), customer nodes, end depot (same as start)
    nodes: List[int]
    customers: List[int]
    vehicles: List[int]
    start: int
    end: int

    coords: Dict[int, Tuple[float, float]]
    dist: Dict[Tuple[int, int], float]
    tt: Dict[Tuple[int, int], float]
    service: Dict[int, float]
    demand: Dict[int, int]
    tw: Dict[int, Tuple[float, float]]
    Q: int
    horizon: float

    # risk coefficients per arc (optional; can be normalized or placeholder)
    risk: Dict[Tuple[int, int], float]

    # mapping to richer objects (optional but helpful)
    node_meta: Dict[int, Node] = field(default_factory=dict)
    vehicle_meta: Dict[int, Vehicle] = field(default_factory=dict)
    shipments: List[Shipment] = field(default_factory=list)


@dataclass
class ExposureMetrics:
    # computed online during simulation
    above_safe_minutes: float = 0.0
    above_critical_minutes: float = 0.0
    max_continuous_excursion_min: float = 0.0
    current_excursion_min: float = 0.0
    cumulative_abuse: float = 0.0  # area above safe_max, weighted by dt


@dataclass
class VehicleSimState:
    vehicle_id: int
    route: List[int]                 # planned sequence from solver (includes start, end)
    route_index: int = 0             # current position in route list
    remaining_dist_to_next: float = 0.0
    current_node: int = 0
    next_node: Optional[int] = None

    # time tracking
    clock_min: float = 0.0
    
    # Multi-compartment tracking
    compartment_temps: Dict[str, float] = field(default_factory=dict)  # {"A": 3.2, "B": 12.1}
    batch_metrics: Dict[int, ExposureMetrics] = field(default_factory=dict)  # Per-batch tracking

    # logs/flags
    delayed_min: float = 0.0
    reroute_triggered: bool = False
