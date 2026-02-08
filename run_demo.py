# run_demo.py
from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

from config import SimConfig
from synthetic_data import build_hub_to_city_instance
from vrptw_solver import solve_vrptw
from sim_engine import simulate_routes
from output_manager import OutputManager
import time


def print_routes(routes: dict[int, list[int]], title: str = "Planned Routes"):
    print(f"\n=== {title} ===")
    for k, r in routes.items():
        print(f"Vehicle {k}: " + " -> ".join(map(str, r)))


def print_reroute_decisions(events):
    """Show only reroute decisions."""
    reroute_events = [e for e in events if e.event == "REROUTE_APPLIED"]
    
    if not reroute_events:
        print("\n=== Reroute Decisions ===")
        print("No reroute decisions made during simulation")
        return
    
    print(f"\n=== Reroute Decisions ({len(reroute_events)} total) ===")
    for i, ev in enumerate(reroute_events, 1):
        print(f"\n--- Decision #{i} ---")
        print(f"Time: {ev.t_min:.1f} min")
        print(f"Vehicle: {ev.vehicle_id}")
        print(f"Trigger reason: {ev.details['reason']}")
        print(f"Batch affected: {ev.details['batch_id']}")
        print(f"\nOPTION CHOSEN: {ev.details['option_selected']}")
        print(f"  Description: {ev.details['option_name']}")
        print(f"  Customers lost: {ev.details['customers_lost']}")
        print(f"  Expected quality gain: {ev.details['expected_quality_gain']:.1%}")
        print(f"  Decision score: {ev.details['score']:.2f}")
        print(f"  New route: {ev.details['new_route']}")


def print_final_routes(inst, res, sim_res):
    """Show actual complete routes taken and customer fulfillment status."""
    print("\n=== Final Routes (Complete Journey) ===")
    
    from monitoring import estimate_quality_remaining
    
    # Extract all customers visited by each vehicle from SERVICE_START events
    visited_by_vehicle = defaultdict(list)
    for event in sim_res.events:
        if event.event == "SERVICE_START":
            node = event.details.get('node')
            if node and node in inst.customers:
                visited_by_vehicle[event.vehicle_id].append(node)
    
    # Print complete journey for each vehicle
    all_visited = set()
    for k in sorted(inst.vehicle_meta.keys()):
        visited = visited_by_vehicle[k]
        all_visited.update(visited)
        
        # Build complete route: depot -> visited customers -> depot
        if visited:
            complete_route = [inst.start] + visited + [inst.end]
            print(f"Vehicle {k}: " + " -> ".join(map(str, complete_route)))
            print(f"  Customers served: {len(visited)}")
        else:
            print(f"Vehicle {k}: {inst.start} -> {inst.end}")
            print(f"  Customers served: 0 (empty route)")
    
    # Build customer -> shipment mapping
    customer_to_shipment = {}
    for shipment in inst.shipments:
        customer_to_shipment[shipment.customer_node_id] = shipment
    
    # Get final quality for each batch
    batch_final_quality = {}
    for vehicle_id, state in sim_res.final_states.items():
        if state.batch_metrics:
            for batch_id, metrics in state.batch_metrics.items():
                # Find the batch
                batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                if batch:
                    quality = estimate_quality_remaining(batch, metrics)
                    batch_final_quality[batch_id] = quality
    
    # Customer fulfillment status
    all_customers = set(inst.customers)
    fulfilled = all_visited & all_customers
    unfulfilled = all_customers - all_visited
    
    print(f"\n=== Customer Fulfillment Status ===")
    print(f"Total customers: {len(all_customers)}")
    
    print(f"\n✅ Fulfilled: {len(fulfilled)} customers")
    if fulfilled:
        for customer_id in sorted(fulfilled):
            shipment = customer_to_shipment.get(customer_id)
            if shipment:
                batch = shipment.batch
                quality = batch_final_quality.get(batch.batch_id, None)
                quality_str = f"{quality:.1%}" if quality is not None else "N/A"
                quality_status = "🟢" if quality and quality >= 0.8 else "🟡" if quality and quality >= 0.6 else "🔴"
                
                print(f"   Customer {customer_id}: {batch.produce_type} ({shipment.demand_units} units) - Quality: {quality_str} {quality_status}")
            else:
                print(f"   Customer {customer_id}: [No shipment data]")
    
    print(f"\n❌ Unfulfilled: {len(unfulfilled)} customers")
    if unfulfilled:
        for customer_id in sorted(unfulfilled):
            shipment = customer_to_shipment.get(customer_id)
            if shipment:
                batch = shipment.batch
                print(f"   Customer {customer_id}: {batch.produce_type} ({shipment.demand_units} units) - NOT DELIVERED")
            else:
                print(f"   Customer {customer_id}: [No shipment data]")
        print(f"   (Abandoned due to rerouting for quality preservation)")
    
    fulfillment_rate = (len(fulfilled) / len(all_customers) * 100) if all_customers else 0
    print(f"\n📊 Fulfillment rate: {fulfillment_rate:.1f}%")
    
    # Quality summary
    if batch_final_quality:
        avg_quality = sum(batch_final_quality.values()) / len(batch_final_quality)
        min_quality = min(batch_final_quality.values())
        print(f"📦 Average delivered quality: {avg_quality:.1%}")
        print(f"📉 Minimum delivered quality: {min_quality:.1%}")



def plot_quality_graphs(inst, sim_res):
    """Plot quality degradation for each product on each truck."""
    print("\n=== Generating Quality Graphs ===")
    
    from monitoring import estimate_quality_remaining
    from data_models import ExposureMetrics
    
    # Group logs by vehicle and extract quality over time
    vehicle_logs = defaultdict(list)
    for row in sim_res.log_rows:
        vehicle_logs[row['vehicle']].append(row)
    
    if not vehicle_logs:
        print("No log data available for plotting")
        return
    
    # Filter out vehicles with no batches
    vehicles_with_batches = {v: logs for v, logs in vehicle_logs.items() 
                             if any('batch_id' in str(log) for log in logs)}
    
    if not vehicles_with_batches:
        # Try to find batch data differently
        vehicles_with_batches = {}
        for v, logs in vehicle_logs.items():
            if any(sim_res.final_states[v].batch_metrics for v in sim_res.final_states if v in vehicle_logs):
                vehicles_with_batches[v] = logs
    
    # Get all vehicles that actually carried shipments
    vehicles_to_plot = []
    for vehicle_id in sorted(vehicle_logs.keys()):
        final_state = sim_res.final_states.get(vehicle_id)
        if final_state and final_state.batch_metrics:
            vehicles_to_plot.append(vehicle_id)
    
    if not vehicles_to_plot:
        print("No vehicles with shipments to plot")
        return
    
    # Create subplots
    num_vehicles = len(vehicles_to_plot)
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(12, 4 * num_vehicles))
    if num_vehicles == 1:
        axes = [axes]
    
    for idx, vehicle_id in enumerate(vehicles_to_plot):
        ax = axes[idx]
        logs = vehicle_logs[vehicle_id]
        final_state = sim_res.final_states[vehicle_id]
        
        # Get all batches for this vehicle
        batch_ids = set(final_state.batch_metrics.keys())
        
        # For each batch, extract quality over time from logs
        for batch_id in sorted(batch_ids):
            # Find the batch object
            batch_obj = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
            if not batch_obj:
                continue
            
            times = []
            qualities = []
            
            # Calculate quality at each logged time point
            # We need to reconstruct metrics from cumulative values in logs
            prev_abuse = 0
            prev_above_safe = 0
            
            for log_row in logs:
                t = log_row['t_min']
                
                # Try to get batch-specific metrics from log if available
                # Otherwise use final metrics (will be constant line)
                if batch_id in final_state.batch_metrics:
                    metrics = final_state.batch_metrics[batch_id]
                    
                    # Estimate proportional progress
                    # Assume linear accumulation over time
                    total_time = logs[-1]['t_min'] if logs else 1
                    progress = t / total_time if total_time > 0 else 1.0
                    
                    # Create metrics at this time point
                    current_metrics = ExposureMetrics(
                        above_safe_minutes=metrics.above_safe_minutes * progress,
                        above_critical_minutes=metrics.above_critical_minutes * progress,
                        max_continuous_excursion_min=metrics.max_continuous_excursion_min * min(1.0, progress * 2),
                        cumulative_abuse=metrics.cumulative_abuse * progress,
                        current_excursion_min=0
                    )
                    
                    quality = estimate_quality_remaining(batch_obj, current_metrics)
                    times.append(t)
                    qualities.append(quality * 100)
            
            if times and qualities:
                label = f"Batch {batch_id} ({batch_obj.produce_type})"
                ax.plot(times, qualities, marker='o', markersize=3, label=label, linewidth=2)
        
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel('Quality Remaining (%)', fontsize=10)
        ax.set_title(f'Vehicle {vehicle_id} - Product Quality Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim([0, 105])
        
        # Add quality zones
        ax.axhspan(80, 100, alpha=0.05, color='green')
        ax.axhspan(60, 80, alpha=0.05, color='yellow')
        ax.axhspan(40, 60, alpha=0.05, color='orange')
        ax.axhspan(0, 40, alpha=0.05, color='red')
    
    plt.tight_layout()
    filename = 'quality_degradation.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Quality graph saved: {filename}")



def plot_temperature_graphs(inst, sim_res):
    """Plot temperature for each compartment on each truck."""
    print("\n=== Generating Temperature Graphs ===")
    
    # Extract temperature data from logs
    temp_data = defaultdict(lambda: defaultdict(list))
    
    for row in sim_res.log_rows:
        vehicle_id = row['vehicle']
        t = row['t_min']
        comp_temps = row.get('comp_temps', {})
        
        for comp_id, temp in comp_temps.items():
            temp_data[vehicle_id][comp_id].append((t, temp))
    
    if not temp_data:
        print("No temperature data available for plotting")
        return
    
    # Create subplot for each vehicle
    num_vehicles = len(temp_data)
    fig, axes = plt.subplots(num_vehicles, 1, figsize=(12, 4 * num_vehicles))
    if num_vehicles == 1:
        axes = [axes]
    
    compartment_colors = {'A': 'blue', 'B': 'green', 'C': 'orange'}
    
    for idx, (vehicle_id, compartments) in enumerate(sorted(temp_data.items())):
        ax = axes[idx]
        
        vehicle = inst.vehicle_meta.get(vehicle_id)
        
        for comp_id, data_points in sorted(compartments.items()):
            if data_points:
                times, temps = zip(*data_points)
                
                # Get setpoint
                setpoint = vehicle.compartments[comp_id].setpoint_c if vehicle else 10.0
                
                color = compartment_colors.get(comp_id, 'gray')
                ax.plot(times, temps, marker='o', label=f'Compartment {comp_id} (Target: {setpoint}°C)',
                       color=color, linewidth=2)
                
                # Plot setpoint line
                ax.axhline(y=setpoint, color=color, linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Vehicle {vehicle_id} - Compartment Temperatures Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    filename = 'compartment_temperatures.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Temperature graph saved: {filename}")


def print_summary(inst, sim_res):
    print("\n=== Final Summary per Vehicle ===")
    for k, st in sim_res.final_states.items():
        print(f"\nVehicle {k}:")
        print(f"  Final node: {st.current_node}")
        print(f"  Delayed by: {st.delayed_min:.1f} min")
        print(f"  Reroute triggered: {st.reroute_triggered}")
        
        # Compartment temperatures
        if st.compartment_temps:
            print(f"  Compartment temperatures:")
            for comp_id, temp in sorted(st.compartment_temps.items()):
                vehicle = inst.vehicle_meta[k]
                setpoint = vehicle.compartments[comp_id].setpoint_c
                print(f"    {comp_id}: {temp:.2f}°C (setpoint: {setpoint:.1f}°C)")
        
        # Per-batch metrics summary
        if st.batch_metrics:
            from monitoring import estimate_quality_remaining
            
            total_above_safe = sum(m.above_safe_minutes for m in st.batch_metrics.values())
            total_abuse = sum(m.cumulative_abuse for m in st.batch_metrics.values())
            
            # Calculate quality for each batch
            qualities = []
            for batch_id, metrics in st.batch_metrics.items():
                batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                if batch:
                    quality = estimate_quality_remaining(batch, metrics)
                    qualities.append(quality)
            
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
            min_quality = min(qualities) if qualities else 0.0
            
            print(f"  Batches tracked: {len(st.batch_metrics)}")
            print(f"  Total above-safe time: {total_above_safe:.1f} min")
            print(f"  Total cumulative abuse: {total_abuse:.2f}")
            print(f"  Average quality: {avg_quality:.1%}")
            print(f"  Minimum quality: {min_quality:.1%}")


def main():
    start_time = time.time()
    cfg = SimConfig()

    # 1) Build synthetic planning instance (hub -> city points)
    print("\n" + "="*70)
    print("MULTI-COMPARTMENT COLD CHAIN SIMULATION WITH AUTONOMOUS REROUTING")
    print("="*70)
    
    inst = build_hub_to_city_instance(
        seed=7,
        n_city_points=12,
        n_vehicles=3,
        vehicle_capacity=20,
        horizon_min=cfg.horizon_min,
        cfg=cfg
    )

    # 2) Solve VRPTW: planned route
    # Using realistic Indian costs: α=₹12/km, β=₹5/min, γ=₹30/risk
    res = solve_vrptw(inst, alpha=12.0, beta=5.0, gamma=30.0, verbose=False)

    if res.status not in [2, 9]:  # 2 OPTIMAL, 9 TIME_LIMIT
        print("No feasible VRPTW solution. Status:", res.status)
        return

    print("\n==============================")
    print("VRPTW PLAN (Gurobi)")
    print("==============================")
    print(f"Objective: {res.obj:.2f}")
    print(f"Total distance: {res.total_dist:.2f}")
    print(f"Total time: {res.total_time:.2f}")
    print(f"Total risk: {res.total_risk:.2f}")

    print_routes(res.routes, "Original Planned Routes (from VRPTW)")

    # 3) Run time-stepped simulation using planned routes
    print("\n" + "="*70)
    print("RUNNING SIMULATION...")
    print("="*70)
    sim_res = simulate_routes(inst, res.routes, cfg, seed=7, enable_reroute_triggers=True)
    print(f"✓ Simulation complete: {len(sim_res.log_rows)} time steps, {len(sim_res.events)} events")

    # 4) Print outputs in order
    print_reroute_decisions(sim_res.events)
    print_final_routes(inst, res, sim_res)
    
    # 5) Generate graphs
    plot_quality_graphs(inst, sim_res)
    plot_temperature_graphs(inst, sim_res)
    
    # 6) Final summary
    print_summary(inst, sim_res)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    
    # 7) Export all results to organized output database
    print("\n📊 Exporting simulation results...")
    output_mgr = OutputManager()
    vrptw_weights = {"alpha": 12.0, "beta": 5.0, "gamma": 30.0}
    execution_time = time.time() - start_time
    
    output_dir = output_mgr.export_all(
        inst=inst,
        vrptw_result=res,
        sim_result=sim_res,
        config=cfg,
        vrptw_weights=vrptw_weights,
        execution_time_sec=execution_time
    )
    
    print(f"\n✅ Results exported to: {output_dir}")
    print(f"   📄 Quick view: {output_dir / 'SUMMARY.txt'}")
    print(f"   📊 Analysis: {output_dir / 'outcomes' / 'deliveries.csv'}")



if __name__ == "__main__":
    import sys
    import subprocess
    import webbrowser
    import time
    from pathlib import Path
    
    # Run the simulation
    main()
    
    # Ask user if they want to launch the dashboard
    print("\n" + "="*70)
    print("🎉 SIMULATION COMPLETE!")
    print("="*70)
    print("\n📊 Would you like to launch the interactive Streamlit dashboard?")
    print("   The dashboard includes:")
    print("   - 🗺️  Animated route visualization")
    print("   - 📈 Quality degradation graphs")
    print("   - 🌡️  Temperature monitoring")
    print("   - 🔄 Reroute decision timeline")
    print("\n" + "="*70)
    
    # Auto-launch option (can be controlled via command line)
    auto_launch = "--dashboard" in sys.argv or "-d" in sys.argv
    
    if auto_launch:
        launch = True
    else:
        response = input("\nLaunch dashboard? (Y/n): ").strip().lower()
        launch = response in ['', 'y', 'yes']
    
    if launch:
        print("\n🚀 Launching Streamlit dashboard...")
        print("   Dashboard will open at: http://localhost:8501")
        print("   Press Ctrl+C in this terminal to stop the dashboard.\n")
        
        # Find streamlit executable
        venv_streamlit = Path(__file__).parent / "venv" / "Scripts" / "streamlit.exe"
        dashboard_file = Path(__file__).parent / "dashboard.py"
        
        if venv_streamlit.exists():
            streamlit_cmd = str(venv_streamlit)
        else:
            streamlit_cmd = "streamlit"  # Try system streamlit
        
        try:
            # Launch streamlit
            subprocess.Popen([streamlit_cmd, "run", str(dashboard_file)], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Open browser
            webbrowser.open("http://localhost:8501")
            
            print("✅ Dashboard launched successfully!")
            print("   If your browser didn't open automatically, navigate to:")
            print("   http://localhost:8501")
            
        except Exception as e:
            print(f"❌ Error launching dashboard: {e}")
            print("\n   You can manually launch it with:")
            print(f"   {streamlit_cmd} run dashboard.py")
    else:
        print("\n✅ Simulation complete! Results saved to:")
        print("   - quality_degradation.png")
        print("   - compartment_temperatures.png")
        print("\n💡 To launch the dashboard later, run:")
        print("   streamlit run dashboard.py")
        print("   OR")
        print("   python run_demo.py --dashboard")

