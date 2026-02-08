"""
output_manager.py

Comprehensive output database manager for cold chain simulation.
Exports results in CSV + JSON + TXT format for easy analysis and sharing.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

from data_models import PlanningInstance, VehicleSimState
from sim_engine import SimulationResult, SimEvent
from vrptw_solver import SolveResult
from config import SimConfig


class OutputManager:
    """Manages all simulation outputs in organized, readable format"""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        self.run_id = None
        self.run_dir = None
        
    def create_run_directory(self, run_id: str = None) -> Path:
        """Create timestamped output directory for this run"""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"run_{timestamp}"
        
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        
        # Create directory structure
        (self.run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "timeseries").mkdir(exist_ok=True)
        (self.run_dir / "outcomes").mkdir(exist_ok=True)
        (self.run_dir / "analysis").mkdir(exist_ok=True)
        (self.run_dir / "analysis" / "visualizations").mkdir(exist_ok=True)
        
        return self.run_dir
    
    def export_all(
        self,
        inst: PlanningInstance,
        vrptw_result: SolveResult,
        sim_result: SimulationResult,
        config: SimConfig,
        vrptw_weights: Dict[str, float],
        execution_time_sec: float
    ):
        """Export all simulation data to organized output"""
        
        if self.run_dir is None:
            self.create_run_directory()
        
        print(f"\n📊 Exporting results to: {self.run_dir}")
        
        # 1. Export metadata
        self._export_config(config, vrptw_weights, execution_time_sec)
        self._export_instance(inst)
        self._export_vrptw_solution(vrptw_result, inst)
        
        # 2. Export time-series data (CSV)
        self._export_vehicle_trajectories(sim_result)
        self._export_temperature_logs(sim_result)
        self._export_quality_logs(sim_result)
        
        # 3. Export outcomes (CSV + JSON)
        self._export_deliveries(inst, sim_result)
        self._export_events(sim_result)
        self._export_reroute_decisions(sim_result)
        
        # 4. Export analysis
        self._export_performance_metrics(inst, sim_result)
        self._export_cost_analysis(inst, sim_result, config)
        
        # 5. Generate summary (TXT)
        self._generate_summary(inst, vrptw_result, sim_result, config, vrptw_weights, execution_time_sec)
        
        print(f"✅ Export complete! View SUMMARY.txt for overview.")
        return self.run_dir
    
    def _export_config(self, config: SimConfig, vrptw_weights: Dict[str, float], execution_time_sec: float):
        """Export simulation configuration"""
        config_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "execution_time_sec": execution_time_sec,
            "vrptw_weights": vrptw_weights,
            "config": {
                "dt_min": config.dt_min,
                "horizon_min": config.horizon_min,
                "ambient_temp_c": config.ambient_temp_c,
                "trigger_min_quality": config.trigger_min_quality,
                "enable_auto_reroute": config.enable_auto_reroute,
                "revenue_per_customer": config.revenue_per_customer,
                "spoilage_cost_per_unit": config.spoilage_cost_per_unit,
            }
        }
        
        with open(self.run_dir / "metadata" / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
    
    def _export_instance(self, inst: PlanningInstance):
        """Export problem instance"""
        instance_data = {
            "n_customers": len(inst.customers),
            "n_vehicles": len(inst.vehicles),
            "vehicle_capacity": inst.Q,
            "horizon_min": inst.horizon,
            "customers": inst.customers,
            "coordinates": {str(k): list(v) for k, v in inst.coords.items()},
            "demands": inst.demand,
            "time_windows": {str(k): list(v) for k, v in inst.tw.items()}
        }
        
        with open(self.run_dir / "metadata" / "instance.json", "w") as f:
            json.dump(instance_data, f, indent=2)
    
    def _export_vrptw_solution(self, result: SolveResult, inst: PlanningInstance):
        """Export VRPTW optimization results"""
        solution_data = {
            "solver_status": result.status,
            "objective_value": result.obj,
            "routes": {f"vehicle_{k}": route for k, route in result.routes.items()},
            "route_metrics": {}
        }
        
        # Calculate per-route metrics
        for k, route in result.routes.items():
            dist = sum(inst.dist.get((route[i], route[i+1]), 0.0) for i in range(len(route)-1))
            time = sum(inst.tt.get((route[i], route[i+1]), 0.0) for i in range(len(route)-1))
            customers = len([n for n in route if n in inst.customers])
            
            solution_data["route_metrics"][f"vehicle_{k}"] = {
                "distance_km": round(dist, 2),
                "time_min": round(time, 2),
                "customers_assigned": customers
            }
        
        with open(self.run_dir / "metadata" / "vrptw_solution.json", "w") as f:
            json.dump(solution_data, f, indent=2)
    
    def _export_vehicle_trajectories(self, sim_result: SimulationResult):
        """Export vehicle trajectories as CSV"""
        csv_path = self.run_dir / "timeseries" / "vehicle_trajectories.csv"
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "vehicle_id", "timestamp_min", "current_node", "next_node",
                "route_index", "delayed_min", "reroute_triggered"
            ])
            
            # Extract from log_rows
            for row in sim_result.log_rows:
                writer.writerow([
                    self.run_id,
                    row.get("vehicle", ""),
                    row.get("t_min", 0.0),
                    row.get("current_node", ""),
                    row.get("next_node", ""),
                    0,  # route_index not in log_rows
                    row.get("delayed_min", 0.0),
                    row.get("reroute_triggered", False)
                ])
    
    def _export_temperature_logs(self, sim_result: SimulationResult):
        """Export temperature logs as CSV"""
        csv_path = self.run_dir / "timeseries" / "temperature_logs.csv"
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "vehicle_id", "compartment_id", "timestamp_min",
                "temperature_c", "setpoint_c", "deviation_c"
            ])
            
            for row in sim_result.log_rows:
                comp_temps = row.get("comp_temps", {})
                if comp_temps:
                    for comp_id, temp in comp_temps.items():
                        # Get setpoint from config (simplified)
                        setpoint = {"A": 3.0, "B": 12.0, "C": 15.0}.get(comp_id, 10.0)
                        writer.writerow([
                            self.run_id,
                            row.get("vehicle", ""),
                            comp_id,
                            row.get("t_min", 0.0),
                            round(temp, 2),
                            setpoint,
                            round(temp - setpoint, 2)
                        ])
    
    def _export_quality_logs(self, sim_result: SimulationResult):
        """Export quality degradation logs as CSV"""
        csv_path = self.run_dir / "timeseries" / "quality_logs.csv"
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "vehicle_id", "batch_id", "timestamp_min",
                "quality_percent", "above_safe_min", "cumulative_abuse"
            ])
            
            for row in sim_result.log_rows:
                batch_qualities = row.get("batch_qualities", {})
                if batch_qualities:
                    for batch_id, quality in batch_qualities.items():
                        # Get abuse metrics from aggregates
                        above_safe = row.get("total_above_safe_min", 0.0)
                        abuse = row.get("total_cumulative_abuse", 0.0)
                        
                        writer.writerow([
                            self.run_id,
                            row.get("vehicle", ""),
                            batch_id,
                            row.get("t_min", 0.0),
                            round(quality * 100, 1),
                            round(above_safe, 1),
                            round(abuse, 1)
                        ])
    
    def _export_deliveries(self, inst: PlanningInstance, sim_result: SimulationResult):
        """Export delivery outcomes as CSV"""
        csv_path = self.run_dir / "outcomes" / "deliveries.csv"
        
        # Build delivery summary
        deliveries = []
        for customer_id in inst.customers:
            # Find if delivered
            delivered = any(
                ev.event == "SERVICE_START" and ev.details.get("node") == customer_id
                for ev in sim_result.events
            )
            
            row = {
                "customer_id": customer_id,
                "status": "DELIVERED" if delivered else "NOT_DELIVERED",
                "product_type": inst.shipments[customer_id-1].batch.produce_type if customer_id <= len(inst.shipments) else "unknown",
                "demand_units": inst.demand.get(customer_id, 0)
            }
            deliveries.append(row)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["customer_id", "status", "product_type", "demand_units"])
            writer.writeheader()
            writer.writerows(deliveries)
    
    def _export_events(self, sim_result: SimulationResult):
        """Export event log as CSV"""
        csv_path = self.run_dir / "outcomes" / "events.csv"
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "timestamp_min", "vehicle_id", "event_type", "details"])
            
            for event in sim_result.events:
                writer.writerow([
                    self.run_id,
                    event.t_min,
                    event.vehicle_id,
                    event.event,
                    json.dumps(event.details)
                ])
    
    def _export_reroute_decisions(self, sim_result: SimulationResult):
        """Export reroute decisions as JSON"""
        reroute_events = [
            ev for ev in sim_result.events
            if ev.event in ["REROUTE_APPLIED", "REROUTE_RECOMMENDATION"]
        ]
        
        decisions = []
        for ev in reroute_events:
            decisions.append({
                "timestamp_min": ev.t_min,
                "vehicle_id": ev.vehicle_id,
                "event_type": ev.event,
                "details": ev.details
            })
        
        with open(self.run_dir / "outcomes" / "reroute_decisions.json", "w") as f:
            json.dump({"decisions": decisions}, f, indent=2)
    
    def _export_performance_metrics(self, inst: PlanningInstance, sim_result: SimulationResult):
        """Export aggregated KPIs as JSON"""
        # Calculate metrics
        total_customers = len(inst.customers)
        service_events = [ev for ev in sim_result.events if ev.event == "SERVICE_START"]
        fulfilled = len(set(ev.details.get("node") for ev in service_events if "node" in ev.details and ev.details.get("node") in inst.customers))
        
        reroute_events = len([ev for ev in sim_result.events if ev.event == "REROUTE_APPLIED"])
        
        metrics = {
            "run_id": self.run_id,
            "total_customers": total_customers,
            "fulfilled_customers": fulfilled,
            "fulfillment_rate": round(fulfilled / total_customers, 3) if total_customers > 0 else 0.0,
            "reroute_events": reroute_events,
            "total_events": len(sim_result.events)
        }
        
        with open(self.run_dir / "analysis" / "performance_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    def _export_cost_analysis(self, inst: PlanningInstance, sim_result: SimulationResult, config: SimConfig):
        """Export cost breakdown as CSV"""
        csv_path = self.run_dir / "analysis" / "cost_breakdown.csv"
        
        # Simplified cost analysis per vehicle
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vehicle_id", "revenue_earned", "fuel_cost", "total_cost", "net_profit"])
            
            for k, state in sim_result.final_states.items():
                # Calculate costs (simplified)
                customers_served = len([n for n in state.route if n in inst.customers])
                revenue = customers_served * config.revenue_per_customer
                
                # Distance from route
                dist = sum(inst.dist.get((state.route[i], state.route[i+1]), 0.0) 
                          for i in range(len(state.route)-1))
                fuel_cost = dist * config.fuel_cost_per_km
                
                writer.writerow([
                    k,
                    round(revenue, 2),
                    round(fuel_cost, 2),
                    round(fuel_cost, 2),
                    round(revenue - fuel_cost, 2)
                ])
    
    def _generate_summary(
        self,
        inst: PlanningInstance,
        vrptw_result: SolveResult,
        sim_result: SimulationResult,
        config: SimConfig,
        vrptw_weights: Dict[str, float],
        execution_time_sec: float
    ):
        """Generate human-readable SUMMARY.txt"""
        
        # Calculate key metrics
        total_customers = len(inst.customers)
        service_events = [ev for ev in sim_result.events if ev.event == "SERVICE_START"]
        fulfilled = len(set(ev.details.get("node") for ev in service_events if "node" in ev.details and ev.details.get("node") in inst.customers))
        fulfillment_rate = fulfilled / total_customers if total_customers > 0 else 0.0
        
        reroute_count = len([ev for ev in sim_result.events if ev.event == "REROUTE_APPLIED"])
        
        # Generate summary
        summary = f"""{'='*80}
COLD CHAIN SIMULATION RESULTS
{'='*80}
Run ID: {self.run_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: ✅ COMPLETED
Runtime: {execution_time_sec:.2f} seconds

CONFIGURATION
{'-'*80}
Customers: {total_customers}    Vehicles: {len(inst.vehicles)}    Capacity: {inst.Q} units
Quality Threshold: {config.trigger_min_quality*100:.0f}%    Rerouting: {'ENABLED' if config.enable_auto_reroute else 'DISABLED'}
Weights: α={vrptw_weights.get('alpha', 1.0)} (distance), β={vrptw_weights.get('beta', 0.0)} (time), γ={vrptw_weights.get('gamma', 0.0)} (risk)

OPTIMIZATION RESULTS
{'-'*80}
✅ Solution Status: {vrptw_result.status} (2=OPTIMAL, 9=TIME_LIMIT)
Objective Value: {vrptw_result.obj:.2f}

SIMULATION OUTCOMES
{'-'*80}
📦 Fulfillment Rate: {fulfillment_rate*100:.1f}% ({fulfilled}/{total_customers} customers)
🔄 Reroute Events: {reroute_count}
📊 Total Events: {len(sim_result.events)}

VEHICLE SUMMARY
{'-'*80}
"""
        
        for k, state in sim_result.final_states.items():
            customers_served = len([n for n in state.route if n in inst.customers])
            dist = sum(inst.dist.get((state.route[i], state.route[i+1]), 0.0) 
                      for i in range(len(state.route)-1))
            
            rerouted_marker = "✅" if state.reroute_triggered else "❌"
            summary += f"Vehicle {k}: {dist:.0f} km, {state.clock_min:.0f} min, {customers_served} customers, Rerouted {rerouted_marker}\n"
        
        summary += f"""
DETAILED DATA
{'-'*80}
📁 Metadata: metadata/config.json, instance.json, vrptw_solution.json
📊 Time-Series: timeseries/vehicle_trajectories.csv, temperature_logs.csv, quality_logs.csv
📋 Outcomes: outcomes/deliveries.csv, events.csv, reroute_decisions.json
📈 Analysis: analysis/performance_metrics.json, cost_breakdown.csv

{'='*80}
"""
        
        with open(self.run_dir / "SUMMARY.txt", "w", encoding="utf-8") as f:
            f.write(summary)
