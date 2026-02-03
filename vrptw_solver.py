# vrptw_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from data_models import PlanningInstance


@dataclass
class SolveResult:
    status: int
    obj: float
    total_dist: float
    total_time: float
    total_risk: float
    routes: Dict[int, List[int]]
    arrival_times: Dict[Tuple[int, int], float]  # (node, vehicle) -> time


def solve_vrptw(
    inst: PlanningInstance,
    alpha: float = 1.0,     # distance weight
    beta: float = 0.0,      # travel time weight
    gamma: float = 0.0,     # temperature-risk weight
    eps_return_time: float = 1e-3,  # tiny tie-breaker to avoid T[end]=horizon
    timelimit: int = 30,
    mipgap: float = 0.01,
    verbose: bool = False
) -> SolveResult:
    """
    VRPTW MILP (Gurobi):
      - each customer served exactly once
      - capacity constraints
      - time windows
      - time propagation with Big-M
      - objective: alpha*distance + beta*time + gamma*risk + eps*sum(T[end,k])

    IMPORTANT FIX:
      - Adds explicit constraints linking end-depot time T[end,k] to the chosen final arc i->end,
        so the printed return time is meaningful.
      - Adds a tiny objective term to make T[end,k] tight (minimal feasible), not floating to horizon.
    """

    N = inst.nodes
    C = inst.customers
    K = inst.vehicles
    s, e = inst.start, inst.end

    m = gp.Model("VRPTW_quality_demo")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = timelimit
    m.Params.MIPGap = mipgap

    # -----------------------------
    # Variables
    # -----------------------------
    x = m.addVars(
        [(i, j, k) for i in N for j in N for k in K if i != j],
        vtype=GRB.BINARY,
        name="x"
    )
    y = m.addVars([(i, k) for i in C for k in K], vtype=GRB.BINARY, name="y")
    use = m.addVars(K, vtype=GRB.BINARY, name="use")
    T = m.addVars(
        [(i, k) for i in N for k in K],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=inst.horizon,
        name="T"
    )

    # -----------------------------
    # Objective
    # -----------------------------
    dist_expr = gp.quicksum(
        inst.dist[(i, j)] * x[i, j, k]
        for i in N for j in N for k in K if i != j
    )
    time_expr = gp.quicksum(
        inst.tt[(i, j)] * x[i, j, k]
        for i in N for j in N for k in K if i != j
    )
    risk_expr = gp.quicksum(
        inst.risk[(i, j)] * x[i, j, k]
        for i in N for j in N for k in K if i != j
    )
    # tiny tie-breaker objective so return time is minimized (becomes "accurate")
    return_expr = gp.quicksum(T[e, k] for k in K)

    m.setObjective(
        alpha * dist_expr + beta * time_expr + gamma * risk_expr + eps_return_time * return_expr,
        GRB.MINIMIZE
    )

    # -----------------------------
    # Constraints
    # -----------------------------

    # (1) Each customer served exactly once
    for i in C:
        m.addConstr(
            gp.quicksum(x[i, j, k] for j in N for k in K if j != i) == 1,
            name=f"serve_{i}"
        )

    # (2) Flow conservation + link y
    for k in K:
        for i in C:
            m.addConstr(
                gp.quicksum(x[j, i, k] for j in N if j != i) ==
                gp.quicksum(x[i, j, k] for j in N if j != i),
                name=f"flow_{i}_{k}"
            )
            m.addConstr(
                gp.quicksum(x[i, j, k] for j in N if j != i) == y[i, k],
                name=f"link_{i}_{k}"
            )

    # (3) Start/end depot behavior if vehicle used
    for k in K:
        m.addConstr(
            gp.quicksum(x[s, j, k] for j in N if j != s) == use[k],
            name=f"depart_{k}"
        )
        m.addConstr(
            gp.quicksum(x[i, e, k] for i in N if i != e) == use[k],
            name=f"arrive_{k}"
        )
        m.addConstr(
            gp.quicksum(x[i, s, k] for i in N if i != s) == 0,
            name=f"no_to_start_{k}"
        )
        m.addConstr(
            gp.quicksum(x[e, j, k] for j in N if j != e) == 0,
            name=f"no_from_end_{k}"
        )

    # (4) Capacity constraints
    for k in K:
        m.addConstr(
            gp.quicksum(inst.demand[i] * y[i, k] for i in C) <= inst.Q,
            name=f"cap_{k}"
        )

    # (NEW) Force all vehicles to be used - each must serve at least 1 customer
    for k in K:
        m.addConstr(
            gp.quicksum(y[i, k] for i in C) >= 1,
            name=f"min_usage_{k}"
        )


    # Big-M
    BIGM = inst.horizon + max(inst.tt.values()) + max(inst.service.values())

    # (5) Time windows
    for k in K:
        m.addConstr(T[s, k] == 0.0, name=f"Tstart_{k}")

        for i in C:
            a, b = inst.tw[i]
            m.addConstr(T[i, k] >= a - BIGM * (1 - y[i, k]), name=f"tw_lb_{i}_{k}")
            m.addConstr(T[i, k] <= b + BIGM * (1 - y[i, k]), name=f"tw_ub_{i}_{k}")

        aE, bE = inst.tw[e]
        m.addConstr(T[e, k] >= aE - BIGM * (1 - use[k]), name=f"tw_end_lb_{k}")
        m.addConstr(T[e, k] <= bE + BIGM * (1 - use[k]), name=f"tw_end_ub_{k}")

    # (6) Time propagation along selected arcs
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                if (i, j, k) not in x:
                    continue
                m.addConstr(
                    T[j, k] >= T[i, k] + inst.service[i] + inst.tt[(i, j)] - BIGM * (1 - x[i, j, k]),
                    name=f"time_{i}_{j}_{k}"
                )

    # (7) End-time linking: ensure T[end,k] matches the chosen final arc i->end
    # Without this, T[end,k] can float. With this + tiny objective, it becomes the true return time.
    for k in K:
        for i in N:
            if i == e:
                continue
            if (i, e, k) in x:
                m.addConstr(
                    T[e, k] >= T[i, k] + inst.service[i] + inst.tt[(i, e)] - BIGM * (1 - x[i, e, k]),
                    name=f"end_link_{i}_{k}"
                )

    # -----------------------------
    # Solve
    # -----------------------------
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return SolveResult(m.Status, float("inf"), 0.0, 0.0, 0.0, {}, {})

    # -----------------------------
    # Extract solution
    # -----------------------------
    routes: Dict[int, List[int]] = {}
    arrival_times: Dict[Tuple[int, int], float] = {(i, k): T[i, k].X for i in N for k in K}

    for k in K:
        if use[k].X < 0.5:
            continue

        succ = {}
        for i in N:
            for j in N:
                if i == j:
                    continue
                if (i, j, k) in x and x[i, j, k].X > 0.5:
                    succ[i] = j

        route = [s]
        cur = s
        safety = 0
        while cur != e and safety < 200:
            cur = succ[cur]
            route.append(cur)
            safety += 1
        routes[k] = route

    # Totals from chosen arcs
    total_dist = 0.0
    total_time = 0.0
    total_risk = 0.0
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                if (i, j, k) in x and x[i, j, k].X > 0.5:
                    total_dist += inst.dist[(i, j)]
                    total_time += inst.tt[(i, j)]
                    total_risk += inst.risk[(i, j)]

    return SolveResult(
        status=m.Status,
        obj=float(m.ObjVal),
        total_dist=total_dist,
        total_time=total_time,
        total_risk=total_risk,
        routes=routes,
        arrival_times=arrival_times
    )
