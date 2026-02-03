"""
Cold Chain Monitoring Dashboard
Interactive Streamlit Dashboard with Route Animation and Real-Time Metrics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dataclasses import asdict
import time

from config import SimConfig
from synthetic_data import build_hub_to_city_instance
from vrptw_solver import solve_vrptw
from sim_engine import simulate_routes
from monitoring import estimate_quality_remaining

# Page configuration
st.set_page_config(
    page_title="Cold Chain Monitoring",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .danger-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def create_route_animation(inst, sim_res):
    """Create map showing vehicle routes"""
    
    # Use actual coordinates from instance
    locations = {}
    for node_id, (x, y) in inst.coords.items():
        locations[node_id] = (x, y)
    
    # Create base map
    fig = go.Figure()
    
    # Add customer markers
    customer_x = [locations[c][0] for c in inst.customers]
    customer_y = [locations[c][1] for c in inst.customers]
    customer_names = [f'Customer {c}' for c in inst.customers]
    
    fig.add_trace(go.Scatter(
        x=customer_x,
        y=customer_y,
        mode='markers+text',
        marker=dict(size=15, color='lightblue', line=dict(width=2, color='darkblue')),
        text=customer_names,
        textposition='top center',
        name='Customers',
        hoverinfo='text'
    ))
    
    # Add depot with smaller marker so it doesn't hide route lines
    depot_x, depot_y = locations[inst.start]
    fig.add_trace(go.Scatter(
        x=[depot_x],
        y=[depot_y],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='darkred')),
        text=['DEPOT'],
        textposition='bottom center',
        textfont=dict(size=14, color='white', family='Arial Black'),
        name='Depot',
        hoverinfo='text',
        hovertext='Depot (Start/End)'
    ))
    
    # Add route lines for each vehicle - COMPLETE JOURNEY FROM DEPOT
    colors = ['#667eea', '#f2994a', '#56ab2f', '#eb3349', '#4facfe']
    
    for k, state in sim_res.final_states.items():
        route = state.route
        
        # DEBUG: Print route to verify it includes depot
        print(f"\n=== Vehicle {k} Route ===")
        print(f"Route nodes: {route}")
        print(f"First node: {route[0]} (should be depot {inst.start})")
        print(f"Last node: {route[-1]} (should be depot {inst.end})")
        
        route_x = []
        route_y = []
        route_labels = []
        
        # Build COMPLETE route: depot → customer1 → customer2 → ... → depot
        for idx, node in enumerate(route):
            if node in locations:
                route_x.append(locations[node][0])
                route_y.append(locations[node][1])
                
                # Label each stop
                if node == inst.start or node == inst.end:
                    route_labels.append(f"DEPOT")
                else:
                    route_labels.append(f"Customer {node}")
            else:
                print(f"WARNING: Node {node} not found in locations!")
        
        if len(route_x) < 2:
            print(f"ERROR: Vehicle {k} has only {len(route_x)} points, cannot draw route!")
            continue
        
        print(f"Drawing {len(route_x)} points for Vehicle {k}")
        print(f"Segments: {len(route_x) - 1}")
        
        # Draw the COMPLETE route line
        fig.add_trace(go.Scatter(
            x=route_x,
            y=route_y,
            mode='lines+markers',
            line=dict(width=3, color=colors[k % len(colors)]),
            marker=dict(
                size=10, 
                color=colors[k % len(colors)],
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            name=f'Vehicle {k}',
            hovertemplate='<b>Vehicle {k}</b><br>%{text}<extra></extra>',
            text=route_labels,
            legendgroup=f'v{k}'
        ))
        
        # Add uniform directional arrows for all segments
        for i in range(len(route_x) - 1):
            fig.add_annotation(
                x=route_x[i+1],
                y=route_y[i+1],
                ax=route_x[i],
                ay=route_y[i],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=colors[k % len(colors)],
                opacity=0.7
            )
    
    # Update layout
    fig.update_layout(
        title='Vehicle Route Visualization',
        xaxis=dict(title='X Coordinate (km)', showgrid=True, zeroline=True),
        yaxis=dict(title='Y Coordinate (km)', showgrid=True, zeroline=True),
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_quality_timeline(inst, sim_res):
    """Create interactive quality degradation timeline"""
    
    fig = make_subplots(
        rows=len(sim_res.final_states), cols=1,
        subplot_titles=[f'Vehicle {k}' for k in sorted(sim_res.final_states.keys())],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, (k, state) in enumerate(sorted(sim_res.final_states.items())):
        if state.batch_metrics:
            for batch_id, metrics in state.batch_metrics.items():
                batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                if batch:
                    # Simple quality timeline (could be enhanced with actual time series)
                    time_points = [0, metrics.above_safe_minutes / 2, metrics.above_safe_minutes]
                    quality_points = [
                        1.0,
                        estimate_quality_remaining(batch, metrics) + 0.15,
                        estimate_quality_remaining(batch, metrics)
                    ]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=quality_points,
                            mode='lines+markers',
                            name=f'{batch.produce_type}',
                            line=dict(color=colors[batch_id % len(colors)], width=3),
                            marker=dict(size=8),
                            hovertemplate='Time: %{x:.0f} min<br>Quality: %{y:.1%}<extra></extra>'
                        ),
                        row=idx+1, col=1
                    )
        
        # Add threshold line
        fig.add_hline(
            y=0.6, line_dash="dash", line_color="red",
            annotation_text="Min Quality", annotation_position="right",
            row=idx+1, col=1
        )
        
        fig.update_yaxes(title_text="Quality %", row=idx+1, col=1, range=[0, 1.05])
        fig.update_xaxes(title_text="Time (min)", row=idx+1, col=1)
    
    fig.update_layout(
        height=300 * len(sim_res.final_states),
        title_text="Product Quality Degradation Over Time",
        showlegend=True
    )
    
    return fig


def create_temperature_heatmap(inst, sim_res):
    """Create temperature heatmap for all compartments"""
    
    # Collect temperature data
    temp_data = []
    for k, state in sim_res.final_states.items():
        for comp_name, temp in state.compartment_temps.items():
            vehicle_meta = inst.vehicle_meta[k]
            setpoint = vehicle_meta.compartments[comp_name].setpoint_c
            temp_data.append({
                'Vehicle': f'V{k}',
                'Compartment': comp_name,
                'Temperature': temp,
                'Setpoint': setpoint,
                'Deviation': temp - setpoint
            })
    
    df = pd.DataFrame(temp_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=df['Compartment'],
        y=df['Vehicle'],
        z=df['Deviation'],
        colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        zmid=0,
        text=df['Temperature'].round(1),
        texttemplate='%{text}°C',
        hovertemplate='Vehicle: %{y}<br>Compartment: %{x}<br>Temp: %{text}°C<extra></extra>',
        colorbar=dict(title="Temp Deviation (°C)")
    ))
    
    fig.update_layout(
        title='Compartment Temperature Status',
        xaxis_title='Compartment',
        yaxis_title='Vehicle',
        height=250
    )
    
    return fig


def display_reroute_timeline(sim_res):
    """Display reroute decisions as interactive timeline"""
    
    reroute_events = [ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"]
    
    if not reroute_events:
        st.info("ℹ️ No rerouting decisions made during simulation")
        return
    
    # Create timeline data
    timeline_data = []
    for ev in reroute_events:
        timeline_data.append({
            'Time': ev.t_min,
            'Vehicle': f'Vehicle {ev.vehicle_id}',
            'Reason': ev.details.get('reason', 'Unknown'),
            'Decision': ev.details.get('option_selected', 'N/A'),
            'Score': ev.details.get('score', 0)
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create timeline figure
    fig = px.scatter(df, x='Time', y='Vehicle', color='Reason', size='Score',
                     hover_data=['Decision', 'Reason', 'Score'],
                     title='Rerouting Decisions Timeline',
                     labels={'Time': 'Simulation Time (min)'})
    
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show details in expandable sections
    for i, event in enumerate(reroute_events):
        with st.expander(f"🔄 Decision #{i+1} - Time: {event.t_min:.1f} min - Vehicle {event.vehicle_id}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Trigger Reason", event.details.get('reason', 'N/A'))
            col2.metric("Option Chosen", event.details.get('option_selected', 'N/A'))
            col3.metric("Decision Score", f"{event.details.get('score', 0):.2f}")


def main():
    # Header
    st.markdown('<div class="main-header">🚚 Cold Chain Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Cold+Chain+AI", use_column_width=True)
        st.markdown("### ⚙️ Simulation Configuration")
        
        n_customers = st.slider("Number of Customers", 5, 20, 12)
        n_vehicles = st.slider("Number of Vehicles", 2, 5, 3)
        vehicle_capacity = st.slider("Vehicle Capacity (units)", 10, 30, 20)
        
        st.markdown("### 🎯 Quality Thresholds")
        min_quality = st.slider("Minimum Quality (%)", 40, 90, 60) / 100
        
        st.markdown("### 💰 Cost Parameters")
        revenue_per_customer = st.number_input("Revenue per Customer (₹)", 500, 2000, 1000)
        spoilage_cost = st.number_input("Spoilage Cost per Unit (₹)", 1000, 5000, 2700)
        
        st.markdown("---")
        run_button = st.button("🚀 Run Simulation", use_container_width=True)
    
    # Main content
    if run_button or 'simulation_run' not in st.session_state:
        with st.spinner("🔄 Running simulation..."):
            # Create config with custom parameters using replace
            from dataclasses import replace
            base_cfg = SimConfig()
            cfg = replace(
                base_cfg,
                trigger_min_quality=min_quality,
                revenue_per_customer=revenue_per_customer,
                spoilage_cost_per_unit=spoilage_cost
            )
            
            # Run simulation
            inst = build_hub_to_city_instance(
                seed=7,
                n_city_points=n_customers,
                n_vehicles=n_vehicles,
                vehicle_capacity=vehicle_capacity,
                horizon_min=cfg.horizon_min,
                cfg=cfg
            )
            
            res = solve_vrptw(inst, verbose=False)
            sim_res = simulate_routes(inst, res.routes, cfg)
            
            # Store in session state
            st.session_state['simulation_run'] = True
            st.session_state['inst'] = inst
            st.session_state['res'] = res
            st.session_state['sim_res'] = sim_res
            st.session_state['cfg'] = cfg
    
    if 'simulation_run' in st.session_state:
        inst = st.session_state['inst']
        res = st.session_state['res']
        sim_res = st.session_state['sim_res']
        cfg = st.session_state['cfg']
        
        # Key Metrics Row
        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        # Calculate metrics
        fulfilled = sum(1 for ev in sim_res.events if ev.event == "SERVICE_START" and ev.details.get('node') in inst.customers)
        total_customers = len(inst.customers)
        fulfillment_rate = (fulfilled / total_customers * 100) if total_customers > 0 else 0
        
        # Average quality
        all_qualities = []
        for state in sim_res.final_states.values():
            if state.batch_metrics:
                for batch_id, metrics in state.batch_metrics.items():
                    batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                    if batch:
                        all_qualities.append(estimate_quality_remaining(batch, metrics))
        avg_quality = (sum(all_qualities) / len(all_qualities) * 100) if all_qualities else 0
        
        # Reroute count
        reroute_count = len([ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"])
        
        # Total distance
        total_dist = res.objective_value if hasattr(res, 'objective_value') else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Fulfillment</h3>
                <h1>{fulfillment_rate:.1f}%</h1>
                <p>{fulfilled}/{total_customers} customers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            quality_card_class = "success-card" if avg_quality >= 80 else "warning-card" if avg_quality >= 60 else "danger-card"
            st.markdown(f"""
            <div class="{quality_card_class}">
                <h3>✨ Avg Quality</h3>
                <h1>{avg_quality:.1f}%</h1>
                <p>All deliveries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 Reroutes</h3>
                <h1>{reroute_count}</h1>
                <p>Decisions made</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Route Animation Section
        st.markdown("### 🗺️ Live Route Visualization")
        route_fig = create_route_animation(inst, sim_res)
        st.plotly_chart(route_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Reroute Timeline
        st.markdown("### 🔄 Rerouting Decisions")
        display_reroute_timeline(sim_res)
        
        st.markdown("---")
        
        # Two column layout for graphs
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📈 Quality Degradation")
            quality_fig = create_quality_timeline(inst, sim_res)
            st.plotly_chart(quality_fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 🌡️ Temperature Status")
            temp_fig = create_temperature_heatmap(inst, sim_res)
            st.plotly_chart(temp_fig, use_container_width=True)
            
            st.markdown("### 📋 Vehicle Summary")
            for k, state in sorted(sim_res.final_states.items()):
                # Calculate distance and time for this vehicle
                route = state.route
                vehicle_dist = 0.0
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    vehicle_dist += inst.dist.get((from_node, to_node), 0.0)
                
                vehicle_time = state.clock_min
                
                with st.expander(f"🚚 Vehicle {k} - {vehicle_dist:.1f} km, {vehicle_time:.0f} min"):
                    # Top metrics
                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Distance", f"{vehicle_dist:.1f} km")
                    met2.metric("Time", f"{vehicle_time:.0f} min")
                    met3.metric("Delayed", f"{state.delayed_min:.1f} min")
                    rerouted = "✅ Yes" if state.reroute_triggered else "❌ No"
                    met4.metric("Rerouted", rerouted)
                    
                    st.write("**Route:**", " → ".join(str(n) for n in route))
                    
                    st.write("**Compartment Temperatures:**")
                    for comp, temp in state.compartment_temps.items():
                        setpoint = inst.vehicle_meta[k].compartments[comp].setpoint_c
                        deviation = temp - setpoint
                        color = "🔴" if abs(deviation) > 3 else "🟡" if abs(deviation) > 1 else "🟢"
                        st.write(f"{color} {comp}: {temp:.2f}°C (setpoint: {setpoint}°C, deviation: {deviation:+.2f}°C)")
        
        st.markdown("---")
        
        # Customer Fulfillment Table
        st.markdown("### 👥 Customer Fulfillment Details")
        
        customer_data = []
        for customer in inst.customers:
            # Check if served
            served = any(ev.event == "SERVICE_START" and ev.details.get('node') == customer for ev in sim_res.events)
            
            # Get shipment info
            shipment = next((s for s in inst.shipments if s.customer_node_id == customer), None)
            if shipment:
                batch = shipment.batch
                product = batch.produce_type
                units = shipment.demand_units
                
                # Get quality if served
                if served:
                    for state in sim_res.final_states.values():
                        if batch.batch_id in state.batch_metrics:
                            quality = estimate_quality_remaining(batch, state.batch_metrics[batch.batch_id])
                            break
                    else:
                        quality = 0
                else:
                    quality = None
                
                customer_data.append({
                    'Customer': f'C{customer}',
                    'Product': product,
                    'Units': units,
                    'Status': '✅ Delivered' if served else '❌ Not Delivered',
                    'Quality': f'{quality:.1%}' if quality is not None else 'N/A'
                })
        
        df_customers = pd.DataFrame(customer_data)
        st.dataframe(df_customers, use_container_width=True, hide_index=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to the Cold Chain Monitoring System</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Configure your simulation parameters in the sidebar and click <strong>Run Simulation</strong> to begin.
            </p>
            <p style="margin-top: 2rem; font-size: 1rem; color: #888;">
                This dashboard provides real-time monitoring of cold chain logistics with:<br>
                🗺️ Animated route visualization | 📊 Quality tracking | 🔄 Autonomous rerouting | 🌡️ Temperature monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
