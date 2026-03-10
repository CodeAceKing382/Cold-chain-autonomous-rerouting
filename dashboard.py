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
import random
from dataclasses import asdict
import time
import requests
import folium
from streamlit_folium import st_folium

from config import SimConfig

@st.cache_data(show_spinner=False)
def get_osrm_route(route_coords):
    """Fetch real road polyline from OSRM between a list of (lat, lon) waypoints."""
    full_poly = []
    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i+1]
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            if data.get("code") == "Ok":
                coords = data["routes"][0]["geometry"]["coordinates"]
                segment_poly = [(lat, lon) for lon, lat in coords]
                if i > 0 and len(segment_poly) > 0:
                    segment_poly = segment_poly[1:]
                full_poly.extend(segment_poly)
            else:
                full_poly.extend([(lat1, lon1), (lat2, lon2)])
        except Exception:
            full_poly.extend([(lat1, lon1), (lat2, lon2)])
    return full_poly

from synthetic_data import build_hub_to_city_instance
from vrptw_solver import solve_vrptw
from sim_engine import simulate_routes
from monitoring import estimate_quality_remaining
from real_geography import REAL_GEOGRAPHY

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



def create_live_folium_map(inst, sim_res, current_sim_time):
    """Create a live Folium map with moving vehicles"""
    
    locations = {}
    for node_id, (lat, lon) in inst.coords.items():
        locations[node_id] = (lat, lon)
    
    city_names = {}
    city_names[inst.start] = REAL_GEOGRAPHY["hub"]["name"]
    for customer_data in REAL_GEOGRAPHY["customers"]:
        city_names[customer_data["id"]] = customer_data["name"]
        
    center_lat = REAL_GEOGRAPHY["hub"]["latitude"]
    center_lon = REAL_GEOGRAPHY["hub"]["longitude"]
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add customers
    for c in inst.customers:
        lat, lon = locations[c]
        folium.Marker(
            [lat, lon],
            popup=city_names.get(c, f'C{c}'),
            tooltip=city_names.get(c, f'C{c}'),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
    # Add Hub
    hub_lat, hub_lon = locations[inst.start]
    folium.Marker(
        [hub_lat, hub_lon],
        popup="Hub: " + city_names.get(inst.start, 'Midnapore'),
        tooltip="Hub",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)
    
    colors = ['blue', 'orange', 'green', 'purple', 'red', 'cadetblue']
    
    # Add Routes and Vehicles
    for k, state in sim_res.final_states.items():
        route = state.route
        route_coords = []
        for node in route:
            if node in locations:
                route_coords.append(locations[node])
                
        if len(route_coords) >= 2:
            # Get real road geometry
            full_poly = get_osrm_route(route_coords)
            if not full_poly:
                full_poly = route_coords
                
            # Draw real road line
            folium.PolyLine(
                full_poly,
                color=colors[k % len(colors)],
                weight=4,
                opacity=0.8,
                tooltip=f'Vehicle {k} Route'
            ).add_to(m)
            
            # Calculate vehicle position at current_sim_time
            max_time = state.clock_min
            if max_time > 0:
                progress = min(current_sim_time / max_time, 1.0)
            else:
                progress = 1.0
                
            segment_count = len(full_poly) - 1
            if segment_count > 0:
                current_segment = int(progress * segment_count)
                if current_segment < segment_count:
                    segment_progress = (progress * segment_count) - current_segment
                    lat1, lon1 = full_poly[current_segment]
                    lat2, lon2 = full_poly[current_segment + 1]
                    vehicle_lat = lat1 + (lat2 - lat1) * segment_progress
                    vehicle_lon = lon1 + (lon2 - lon1) * segment_progress
                else:
                    vehicle_lat, vehicle_lon = full_poly[-1]
                    
                folium.Marker(
                    [vehicle_lat, vehicle_lon],
                    popup=f"Vehicle {k}",
                    tooltip=f"Vehicle {k} (Time: {current_sim_time:.1f}m)",
                    icon=folium.Icon(color=colors[k % len(colors)], icon='truck', prefix='fa')
                ).add_to(m)
                
    return m


def create_quality_timeline(inst, sim_res, current_time=None):
    """Create interactive quality degradation timeline"""
    
    fig = make_subplots(
        rows=len(sim_res.final_states), cols=1,
        subplot_titles=[f'Vehicle {k}' for k in sorted(sim_res.final_states.keys())],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    # Create stable dictionary mapping produce_type -> color to prevent mismatch
    produce_color_map = {}
    color_idx = 0
    
    for idx, (k, state) in enumerate(sorted(sim_res.final_states.items())):
        vehicle_time = state.clock_min
        if vehicle_time <= 0:
            vehicle_time = 1.0
            
        if state.batch_metrics:
            for batch_id, metrics in state.batch_metrics.items():
                batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                if batch:
                    # Original formula structure
                    original_times = [
                        0, 
                        metrics.above_safe_minutes / 2 if hasattr(metrics, 'above_safe_minutes') else vehicle_time / 2, 
                        metrics.above_safe_minutes if hasattr(metrics, 'above_safe_minutes') else vehicle_time
                    ]
                    
                    original_qualities = [
                        1.0,
                        estimate_quality_remaining(batch, metrics) + 0.15,
                        estimate_quality_remaining(batch, metrics)
                    ]
                    
                    # Truncate to current sim time
                    time_points = []
                    quality_points = []
                    
                    cutoff_time = current_time if current_time is not None else vehicle_time
                    
                    for i in range(len(original_times)):
                        if original_times[i] <= cutoff_time:
                            time_points.append(original_times[i])
                            quality_points.append(original_qualities[i])
                        else:
                            # Interpolate the cutoff point
                            if i > 0:
                                prev_t = original_times[i-1]
                                next_t = original_times[i]
                                prev_q = original_qualities[i-1]
                                next_q = original_qualities[i]
                                
                                fraction = (cutoff_time - prev_t) / (next_t - prev_t) if next_t > prev_t else 0
                                interp_q = prev_q + (next_q - prev_q) * fraction
                                
                                time_points.append(cutoff_time)
                                quality_points.append(interp_q)
                            break
                    
                    # Ensure consistent color mapping for this specific produce type
                    p_type = batch.produce_type
                    if p_type not in produce_color_map:
                        produce_color_map[p_type] = colors[color_idx % len(colors)]
                        color_idx += 1
                        
                    # Track if we already added a legend entry for this product globally
                    show_in_legend = (p_type not in [t.name for t in fig.data])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=quality_points,
                            mode='lines+markers',
                            name=f'{p_type}',
                            legendgroup=f'{p_type}',
                            showlegend=show_in_legend,
                            line=dict(color=produce_color_map[p_type], width=3),
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
        
        # Fix the x-axis relative to the full journey so the graph visually "fills up" instead of zooming
        fig.update_yaxes(title_text="Quality %", row=idx+1, col=1, range=[0, 1.05])
        fig.update_xaxes(title_text="Time (min)", row=idx+1, col=1, range=[0, vehicle_time * 1.05])
    
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


def display_reroute_timeline(sim_res, current_time=None):
    """Display reroute decisions as interactive timeline"""
    
    reroute_events = [ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"]
    if current_time is not None:
        reroute_events = [ev for ev in reroute_events if ev.t_min <= current_time]
        
    if not reroute_events:
        st.info("ℹ️ No rerouting decisions made yet")
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
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 1rem; border-radius: 10px; text-align: center;
                    color: white; font-size: 1.2rem; font-weight: bold;
                    letter-spacing: 1px; margin-bottom: 0.5rem;">
            🚚 Cold Chain AI
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚙️ Simulation Configuration")
        
        # Fixed number of customers (12 real cities in Medinipur)
        n_customers = 12
        st.info(f"📍 **{n_customers} Customer Cities** (Medinipur District)")
        
        n_vehicles = st.slider("Number of Vehicles", 2, 5, 3)
        vehicle_capacity = st.slider("Vehicle Capacity (units)", 10, 30, 20)
        
        st.markdown("### 📦 Customer Demands")
        st.caption("Set demand for each city (crates/units)")
        
        # Individual demand sliders for each customer
        customer_demands = {}
        from real_geography import REAL_GEOGRAPHY
        
        # Create 2 columns for better layout
        col1, col2 = st.columns(2)
        
        for idx, customer in enumerate(REAL_GEOGRAPHY["customers"]):
            city_name = customer["name"]
            customer_id = customer["id"]
            
            # Alternate between columns
            with col1 if idx % 2 == 0 else col2:
                customer_demands[customer_id] = st.number_input(
                    f"{city_name}",
                    min_value=1,
                    max_value=10,
                    value=random.randint(1, 4),
                    step=1,
                    key=f"demand_{customer_id}"
                )
        
        st.markdown("### 🎯 Quality Thresholds")
        min_quality = st.slider("Minimum Quality (%)", 40, 90, 60) / 100
        
        st.markdown("### ⚖️ VRPTW Optimization Weights")
        st.caption("Economic costs: Distance (₹/km), Time (₹/min), Risk (₹/unit)")
        alpha_weight = st.slider("Distance Cost α (₹/km)", 5.0, 20.0, 12.0, 0.5)
        beta_weight = st.slider("Time Cost β (₹/min)", 2.0, 10.0, 5.0, 0.5)
        gamma_weight = st.slider("Risk Cost γ (₹/unit)", 10.0, 50.0, 30.0, 1.0)
        
        st.markdown("### 💰 Economic Parameters")
        
        # Revenue
        st.markdown("**💵 Revenue**")
        revenue_per_customer = st.number_input("Revenue per Customer (₹)", 500, 2000, 1000, key="revenue")
        
        # Operating Costs
        st.markdown("**� Operating Costs**")
        col1, col2 = st.columns(2)
        with col1:
            fuel_cost_per_km = st.number_input("Fuel (₹/km)", 5, 20, 12, key="fuel")
        with col2:
            driver_wage_per_hour = st.number_input("Driver (₹/hr)", 50, 200, 100, key="driver")
        
        col3, col4 = st.columns(2)
        with col3:
            vehicle_rental_per_day = st.number_input("Vehicle Rental (₹/day)", 1000, 5000, 2000, key="rental")
        with col4:
            refrigeration_cost_per_hour = st.number_input("Refrigeration (₹/hr)", 20, 100, 50, key="refrig")
        
        # Quality & Penalty Costs
        st.markdown("**⚠️ Quality & Penalties**")
        col5, col6 = st.columns(2)
        with col5:
            spoilage_cost = st.number_input("Spoilage (₹/unit)", 1000, 5000, 2700, key="spoilage")
        with col6:
            temp_violation_penalty = st.number_input("Temp Violation (₹)", 100, 1000, 500, key="temp_penalty")
        
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
                cfg=cfg,
                custom_demands=customer_demands  # Pass custom demands
            )
            
            res = solve_vrptw(inst, alpha=alpha_weight, beta=beta_weight, gamma=gamma_weight, verbose=False)
            sim_res = simulate_routes(inst, res.routes, cfg)
            
            # Store in session state
            st.session_state['simulation_run'] = True
            st.session_state['last_real_time'] = time.time()
            st.session_state['current_sim_time'] = 0.0
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
        
        
        # Calculate max simulation time first
        max_sim_time = 0
        if sim_res.final_states:
            max_sim_time = max(state.clock_min for state in sim_res.final_states.values())
            
        # Add fast forward control at the top level
        st.markdown("### 🗺️ Live Route Visualization")
        speed_multiplier = st.radio(
            "⏩ Simulation Speed", 
            options=[1, 5, 10, 30, 60], 
            format_func=lambda x: "1x (Real Time)" if x == 1 else f"{x}x Fast Forward",
            horizontal=True
        )
        
        # Time calculation
        current_real_time = time.time()
        if 'last_real_time' not in st.session_state:
            st.session_state['last_real_time'] = current_real_time
            st.session_state['current_sim_time'] = 0.0
            
        delta_real_seconds = current_real_time - st.session_state['last_real_time']
        st.session_state['last_real_time'] = current_real_time
        
        sim_minutes_delta = (delta_real_seconds / 60.0) * speed_multiplier
        st.session_state['current_sim_time'] += sim_minutes_delta
        current_sim_time = min(st.session_state['current_sim_time'], max_sim_time)
        
        # Progress Bar
        progress_pct = min(100.0, (current_sim_time / max_sim_time * 100) if max_sim_time > 0 else 100.0)
        st.progress(progress_pct / 100.0, text=f"Simulation Time: {current_sim_time:.1f} min / {max_sim_time:.1f} min")
        
        # Calculate real-time metrics up to current_sim_time
        fulfilled = sum(1 for ev in sim_res.events if ev.event == "SERVICE_START" and ev.details.get('node') in inst.customers and ev.t_min <= current_sim_time)
        total_customers = len(inst.customers)
        fulfillment_rate = (fulfilled / total_customers * 100) if total_customers > 0 else 0
        
        # Average quality (simplified to just use final state but scale based on time, true tracking would require full event log)
        # Using final states as approximation since quality only drops
        all_qualities = []
        for state in sim_res.final_states.values():
            if state.batch_metrics:
                for batch_id, metrics in state.batch_metrics.items():
                    batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                    if batch:
                        # Estimate current quality based on fraction of trip complete
                        fraction = min(1.0, current_sim_time / state.clock_min) if state.clock_min > 0 else 1.0
                        start_q = 1.0
                        end_q = estimate_quality_remaining(batch, metrics)
                        current_q = start_q - (start_q - end_q) * fraction
                        all_qualities.append(current_q)
        avg_quality = (sum(all_qualities) / len(all_qualities) * 100) if all_qualities else 100.0
        
        # Reroute count up to current time
        reroute_count = len([ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED" and ev.t_min <= current_sim_time])
        
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
        
        route_map = create_live_folium_map(inst, sim_res, current_sim_time)
        st_folium(route_map, use_container_width=True, height=600, returned_objects=[])
        
        st.markdown("---")
        
        # Reroute Timeline
        st.markdown("### 🔄 Rerouting Decisions")
        display_reroute_timeline(sim_res, current_sim_time)
        
        st.markdown("---")
        
        # Two column layout for graphs
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📈 Quality Degradation")
            quality_fig = create_quality_timeline(inst, sim_res, current_sim_time)
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
                
                with st.expander(f"🚚 Vehicle {k} - {vehicle_dist:.1f} km, {vehicle_time:.0f} min (Final Expected)"):
                    
                    # Vehicle status based on current time
                    status = "Finished" if current_sim_time >= vehicle_time else "En Route" if current_sim_time > 0 else "Waiting"
                    
                    # Top metrics
                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Status", status)
                    met2.metric("Est. Total Time", f"{vehicle_time:.0f} min")
                    met3.metric("Delayed", f"{state.delayed_min:.1f} min")
                    
                    rerouted = "✅ Yes" if any(ev.t_min <= current_sim_time for ev in sim_res.events if ev.event == "REROUTE_APPLIED" and ev.vehicle_id == k) else "❌ No"
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
            # Check if served up to current time
            served = any(ev.event == "SERVICE_START" and ev.details.get('node') == customer and ev.t_min <= current_sim_time for ev in sim_res.events)
            
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
        
        # Trigger rerun if simulation is still ongoing
        if current_sim_time < max_sim_time:
            time.sleep(10)  # Wait 10 seconds between updates as requested
            st.rerun()
    
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
