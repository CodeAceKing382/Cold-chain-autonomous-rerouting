# synthetic_data.py
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

from data_models import Node, Vehicle, ProduceBatch, Shipment, PlanningInstance
from config import SimConfig


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_hub_to_city_instance(
    seed: int = 7,
    n_city_points: int = 12,
    n_vehicles: int = 3,
    vehicle_capacity: int = 20,
    horizon_min: int = 300,
    cfg: SimConfig = None,
) -> PlanningInstance:
    """
    Generates a usable synthetic VRPTW instance:
      - 1 hub (depot)
      - n_city_points customer delivery points (mandi/DC/retail)
      - n_vehicles identical-ish vehicles
      - demands, time windows, service times
      - placeholder arc risk (later replaced by simulated monitoring-based risk)

    TODO hooks:
      - replace time windows with real mandi opening/receiving windows
      - replace travel times with OSRM / real road network
      - replace risk with temperature-abuse driven prediction
    """
    if cfg is None:
        cfg = SimConfig()

    random.seed(seed)

    # --- Node indexing ---
    start = 0
    end = n_city_points + 1
    nodes = list(range(end + 1))
    customers = list(range(1, n_city_points + 1))
    vehicles = list(range(n_vehicles))
    Q = vehicle_capacity
    horizon = float(horizon_min)

    # --- Place hub in center; customers scattered ---
    node_meta: Dict[int, Node] = {}
    coords: Dict[int, Tuple[float, float]] = {}

    hub_x, hub_y = 50.0, 50.0
    coords[start] = (hub_x, hub_y)
    coords[end] = (hub_x, hub_y)

    node_meta[start] = Node(
        node_id=start, node_type="hub", name="Rural Hub",
        x=hub_x, y=hub_y, tw_open=0.0, tw_close=horizon, service_min=0.0
    )
    node_meta[end] = Node(
        node_id=end, node_type="hub", name="Rural Hub (return)",
        x=hub_x, y=hub_y, tw_open=0.0, tw_close=horizon, service_min=0.0
    )

    # customer nodes
    for i in customers:
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)

        # TODO: replace with observed receiving windows (mandi/DC)
        # Here we create "plausible" windows: open near earliest arrival, close later.
        # We'll set service time 5-10 min.
        service_min = float(random.choice([5, 5, 5, 8, 10]))

        coords[i] = (x, y)

        node_meta[i] = Node(
            node_id=i,
            node_type="city_point",
            name=f"City Drop {i}",
            x=x, y=y,
            tw_open=0.0,    # temp placeholder, overwritten below
            tw_close=horizon,
            service_min=service_min
        )

    # --- Distance / travel time matrices ---
    dist: Dict[Tuple[int, int], float] = {}
    tt: Dict[Tuple[int, int], float] = {}

    # TODO: replace speed model with road-network travel times (OSRM/Google)
    # For now: travel_time = distance / speed_factor. Keep speed_factor ~1.
    base_speed = 1.33  # km/min (80 km/h - highway speed)

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            d = euclid(coords[i], coords[j])
            dist[(i, j)] = d
            tt[(i, j)] = d / base_speed

    # --- Service time dict ---
    service: Dict[int, float] = {i: node_meta[i].service_min for i in nodes}

    # --- Demands ---
    demand: Dict[int, int] = {i: 0 for i in nodes}
    for i in customers:
        # TODO: replace with real order size distributions (crates/kg)
        demand[i] = random.randint(1, 4)

    # --- Time windows ---
    tw: Dict[int, Tuple[float, float]] = {start: (0.0, horizon), end: (0.0, horizon)}

    for i in customers:
        earliest_arrival = tt[(start, i)]  # naive
        open_t = max(0.0, earliest_arrival - 20.0)
        close_t = min(horizon, earliest_arrival + 120.0)
        if close_t - open_t < 60:
            close_t = min(horizon, open_t + 80.0)

        tw[i] = (open_t, close_t)

        # also update node_meta for convenience
        node_meta[i].tw_open = open_t
        node_meta[i].tw_close = close_t

    # --- Vehicles metadata (multi-compartment) ---
    vehicle_meta: Dict[int, Vehicle] = {}
    for k in vehicles:
        # Create compartments based on config specs
        from data_models import Compartment
        compartments = {}
        
        for comp_id, spec in cfg.compartment_specs.items():
            comp_capacity = int(Q * spec["capacity_fraction"])
            compartments[comp_id] = Compartment(
                compartment_id=comp_id,
                setpoint_c=spec["setpoint_c"],
                capacity=comp_capacity,
                temp_c=spec["setpoint_c"],  # Start at setpoint
                cooling_efficiency=random.uniform(0.85, 0.95),  # Better efficiency
                insulation_quality=0.7  # Improved insulation (lower = better)
            )
        
        vehicle_meta[k] = Vehicle(
            vehicle_id=k,
            capacity=Q,
            max_speed_factor=random.uniform(0.9, 1.1),
            compartments=compartments
        )

    # --- Produce batches + shipments assignment ---
    # We attach one shipment to each customer (simple baseline).
    shipments: List[Shipment] = []
    batch_id = 0
    shipment_id = 0

    produce_types = list(cfg.produce_catalog.keys())
    for i in customers:
        p = random.choice(produce_types)
        params = cfg.produce_catalog[p]

        # TODO: replace with real priority logic (e.g., leafy/milk high priority)
        priority = 1 if p in ["milk", "leafy", "flowers"] else 2

        batch = ProduceBatch(
            batch_id=batch_id,
            produce_type=p,
            safe_min_c=params["safe"][0],
            safe_max_c=params["safe"][1],
            critical_c=params["critical"],
            shelf_life_h=params["shelf_life_h"],
            k_abuse=params["k_abuse"],
            priority=priority
        )
        batch_id += 1
        
        # Assign to appropriate compartment based on product type
        assigned_comp = cfg.product_to_compartment[p]

        shipments.append(
            Shipment(
                shipment_id=shipment_id,
                customer_node_id=i,
                demand_units=demand[i],
                batch=batch,
                assigned_compartment=assigned_comp  # Assign compartment
            )
        )
        shipment_id += 1

    # --- Placeholder risk on arcs (planning-time proxy) ---
    # NOTE: In later phases, risk should be derived from monitoring/simulation.
    road_badness = {(i, j): random.uniform(0.8, 1.6) for (i, j) in dist.keys()}
    ambient_hot = 1.0  # TODO: replace with ambient_temp profile feature
    risk: Dict[Tuple[int, int], float] = {}
    for (i, j), tij in tt.items():
        risk[(i, j)] = tij * (0.6 * road_badness[(i, j)] + 0.4 * ambient_hot)

    return PlanningInstance(
        nodes=nodes,
        customers=customers,
        vehicles=vehicles,
        start=start,
        end=end,
        coords=coords,
        dist=dist,
        tt=tt,
        service=service,
        demand=demand,
        tw=tw,
        Q=Q,
        horizon=horizon,
        risk=risk,
        node_meta=node_meta,
        vehicle_meta=vehicle_meta,
        shipments=shipments
    )
