# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
from pathlib import Path

# Debug: Print current working directory and files
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Add src directory to path
src_path = str(Path(__file__).parent / 'src')
print("\nAttempting to add to path:", src_path)
print("Current sys.path:", sys.path)

if os.path.exists(src_path):
    print("\nContents of src directory:", os.listdir(src_path))
    if os.path.exists(os.path.join(src_path, 'agents')):
        print("\nContents of src/agents:", os.listdir(os.path.join(src_path, 'agents')))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import agents
try:
    print("\nAttempting to import from agents...")
    from agents.agent_registry import get_agent, REGISTRY
    from agents.agent_base import BaseAgent
    print("Successfully imported agents!")
except ImportError as e:
    print(f"\nError importing agents: {e}")
    print(f"Current sys.path: {sys.path}")
    st.error(f"Failed to import agents: {e}")
    st.error(f"Current sys.path: {sys.path}")
    if os.path.exists(src_path):
        st.error(f"Contents of {src_path}: {os.listdir(src_path)}")
    raise

# Set page config
st.set_page_config(
    page_title="AirTempTS - Agent Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .agent-card { 
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: #f8f9fa;
    }
    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .agent-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: capitalize;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = {}
    st.session_state.results = {}
    st.session_state.data_loaded = False

# Load sample data
def load_sample_data():
    # Generate sample temperature data if needed
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    temps = 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365)
    return pd.DataFrame({
        'date': dates,
        'mean_temp': temps
    })

def get_temp_column(df):
    """Get the temperature column name from DataFrame"""
    possible_names = ['mean_temp', 'temperature', 'temp', 'Temperature', 'Mean_Temp']
    for col in possible_names:
        if col in df.columns:
            return col
    # If none found, return first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    raise ValueError("No suitable temperature column found")

# Main App
def main():
    st.title("🌡️ AirTempTS - Agent Dashboard")
    
    # Load data
    if not st.session_state.data_loaded:
        st.session_state.df = load_sample_data()
        st.session_state.data_loaded = True
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Agents", "Results", "Data"])
    
    if page == "Agents":
        show_agents_page()
    elif page == "Results":
        show_results_page()
    elif page == "Data":
        show_data_page()

def show_agents_page():
    st.header("Available Agents")
    
    # Agent descriptions
    agent_descriptions = {
        "resonant": "Performs spectral analysis to detect dominant frequencies in temperature data",
        "wavelet": "Identifies transient features using wavelet transforms",
        "alchemist": "Engineers features from raw temperature data",
        "bayes_fusion": "Combines multiple data sources using Bayesian methods",
        "uncertainty": "Quantifies uncertainty in temperature predictions",
        "changepoint": "Detects regime changes in temperature patterns",
        "tesla_osc": "Identifies oscillatory patterns in temperature data"
    }
    
    # Display agent cards
    for agent_name, agent_class in REGISTRY.items():
        with st.container():
            st.markdown(f"""
                <div class="agent-card">
                    <div class="agent-header">
                        <h3 class="agent-title">{agent_name.title()} Agent</h3>
                        <span class="status-badge" style="background: #e6f7ff; color: #1890ff;">
                            {agent_name}
                        </span>
                    </div>
                    <p>{agent_descriptions.get(agent_name, "No description available")}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add run button
            if st.button(f"Run {agent_name.title()} Agent", key=f"run_{agent_name}"):
                run_agent(agent_name)
    
    st.markdown("---")
    st.subheader("Agent Execution Log")
    if 'execution_log' in st.session_state:
        for log in reversed(st.session_state.execution_log):
            st.code(log, language="python")

def run_agent(agent_name):
    """Execute an agent and store results"""
    if 'df' not in st.session_state or not st.session_state.data_loaded:
        st.error("Please load data first on the Data page")
        return
    
    df = st.session_state.df
    
    try:
        # Get the agent instance
        agent = get_agent(agent_name)
        
        # Prepare the data - ensure it has the expected column name
        working_df = df.copy()
        temp_col = get_temp_column(working_df)
        if temp_col != 'mean_temp':
            working_df = working_df.rename(columns={temp_col: 'mean_temp'})
        
        # Run the agent
        with st.spinner(f"Running {agent_name}..."):
            try:
                # For UncertaintySynthesisAgent, we need to handle it specially
                if agent_name == "uncertainty":
                    # Initialize the agent with the data
                    try:
                        # Check if the agent has a fit method and call it if it exists
                        if hasattr(agent, 'fit') and callable(agent.fit):
                            agent.fit(working_df)
                            
                        # Create a proper result dictionary with all required fields
                        result = {
                            'agent': agent,
                            'status': 'success',
                            'message': f"{agent_name} agent initialized successfully",
                            'transformed_data': working_df,  # Include the original data
                            'agent_type': 'UncertaintySynthesisAgent'
                        }
                        
                        # If the agent has a plot method, include a simple visualization
                        if hasattr(agent, 'plot') and callable(agent.plot):
                            try:
                                fig = agent.plot()
                                if fig is not None:
                                    result['plot'] = fig
                            except Exception as e:
                                st.warning(f"Could not generate initial plot: {str(e)}")
                                
                    except Exception as e:
                        result = {
                            'agent': agent,
                            'status': 'warning',
                            'message': f"{agent_name} agent loaded but encountered an error: {str(e)}",
                            'transformed_data': working_df
                        }
                
                # For other agents, use the standard approach
                else:
                    # Check for required agent methods
                    has_run = hasattr(agent, 'run') and callable(agent.run)
                    has_transform = (hasattr(agent, 'transform') and 
                                   callable(agent.transform) and 
                                   not hasattr(agent.transform, "__isabstractmethod__"))
                    has_fit = (hasattr(agent, 'fit') and 
                              callable(agent.fit) and 
                              not hasattr(agent.fit, "__isabstractmethod__"))
                    
                    # Execute based on available methods
                    if has_run:
                        result = agent.run(working_df)
                    elif has_transform:
                        # Standard agent interface - check for fit/transform pattern first
                        if has_fit:
                            try:
                                # First try to fit the agent if it's not already fitted
                                if not getattr(agent, '_is_fitted', False):
                                    agent.fit(working_df)
                                # Transform the data
                                transformed = agent.transform(working_df)
                                result = {
                                    'transformed_data': transformed,
                                    'agent': agent,
                                    'message': f"{agent_name} transformed data successfully"
                                }
                            except Exception as e:
                                st.warning(f"Fit/transform failed: {str(e)}. Trying transform only...")
                                # If fit/transform fails, try transform only
                                try:
                                    transformed = agent.transform(working_df)
                                    result = {
                                        'transformed_data': transformed,
                                        'agent': agent,
                                        'message': f"{agent_name} transformed data successfully (fit skipped)"
                                    }
                                except Exception as e2:
                                    st.error(f"Transform failed: {str(e2)}")
                                    raise
                        else:
                            # Transform only
                            transformed = agent.transform(working_df)
                            result = {
                                'transformed_data': transformed,
                                'agent': agent,
                                'message': f"{agent_name} transformed data successfully"
                            }
                    # For callable agents
                    elif callable(agent):
                        result = agent(working_df)
                    # For agents that might have custom methods
                    else:
                        # Check for other callable methods that might be the main entry point
                        callable_methods = [
                            (name, method) for name, method in agent.__class__.__dict__.items()
                            if callable(method) and not name.startswith('_') and name not in ['fit', 'transform', 'predict']
                        ]
                        if callable_methods:
                            # Try each method until one works
                            for method_name, method in callable_methods:
                                try:
                                    # Check if this is a method that requires self
                                    if method.__code__.co_varnames and method.__code__.co_varnames[0] == 'self':
                                        result = method(agent, working_df)
                                    else:
                                        result = method(working_df)
                                    # If we get here without error, this method worked
                                    st.info(f"Used method '{method_name}' for {agent_name}")
                                    break
                                except Exception as e:
                                    continue
                            else:
                                # If we exhausted all methods and none worked
                                raise AttributeError(
                                    f"Agent {agent_name} does not have a valid interface. "
                                    f"Tried methods: {[m[0] for m in callable_methods]}"
                                )
                        else:
                            raise AttributeError(
                                f"Agent {agent_name} does not have a valid interface. "
                                f"Expected one of: fit()/transform(), run(), or callable"
                            )
                
                # Ensure we have a result
                if 'result' not in locals():
                    result = {
                        'agent': agent_name,
                        'status': 'completed',
                        'message': 'Agent executed but no result was returned'
                    }
                    
            except Exception as e:
                st.error(f"Error executing agent {agent_name}: {str(e)}")
                # Include more detailed error information
                import traceback
                st.code(traceback.format_exc(), language='python')
                # Re-raise to stop execution
                raise
        
        # Store the results
        if 'results' not in st.session_state:
            st.session_state.results = {}
        st.session_state.results[agent_name] = result
        
        # Store the agent in the agents dictionary
        if 'agents' not in st.session_state:
            st.session_state.agents = {}
        st.session_state.agents[agent_name] = agent
        
        st.success(f"{agent_name} completed successfully!")
        
    except Exception as e:
        st.error(f"Error running {agent_name}: {str(e)}")
        st.exception(e)  # Show full traceback for debugging

def show_results_page():
    st.header("Agent Results")
    
    if not st.session_state.results:
        st.info("No agent results available. Run some agents first.")
        return
    
    # Display results for each agent
    for agent_name, result in st.session_state.results.items():
        with st.expander(f"Results from {agent_name}", expanded=True):
            try:
                # Handle different result formats
                if result is None:
                    st.warning("No results returned from this agent")
                    continue
                    
                # Handle fit/transform results
                if isinstance(result, dict) and 'transformed_data' in result:
                    st.subheader("Transformation Results")
                    st.write(result.get('message', 'Transformation completed'))
                    
                    # Display the transformed data if it's a DataFrame
                    transformed_data = result.get('transformed_data')
                    if hasattr(transformed_data, 'head'):  # Check if it's a DataFrame
                        # Show data summary
                        st.write("### Data Preview")
                        st.dataframe(transformed_data.head())
                        
                        # Show statistics
                        st.write("### Statistics")
                        st.dataframe(transformed_data.describe())
                        
                        # Plot numerical columns
                        try:
                            import plotly.express as px
                            from plotly.subplots import make_subplots
                            import plotly.graph_objects as go
                            
                            num_cols = transformed_data.select_dtypes(include=['float64', 'int64']).columns
                            
                            if len(num_cols) > 0:
                                st.write("### Time Series Plots")
                                # Create tabs for each numerical column
                                tabs = st.tabs([str(col) for col in num_cols])
                                
                                for tab, col in zip(tabs, num_cols):
                                    with tab:
                                        fig = px.line(transformed_data, y=col,
                                                    title=f"{col} over time",
                                                    labels={'value': str(col), 'index': 'Time'})
                                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{agent_name}_{col}")
                        
                        except Exception as e:
                            st.warning(f"Could not generate all plots: {str(e)}")
                    
                    # Show agent-specific visualizations
                    if 'agent' in result and hasattr(result['agent'], 'plot') and callable(result['agent'].plot):
                        st.write("### Model Visualization")
                        try:
                            fig = result['agent'].plot()
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate model visualization: {str(e)}")
                    
                    # Show agent summary if available
                    if 'agent' in result and hasattr(result['agent'], 'get_summary') and callable(result['agent'].get_summary):
                        with st.expander("Agent Summary", expanded=False):
                            try:
                                summary = result['agent'].get_summary()
                                if isinstance(summary, dict):
                                    st.json(summary)
                                else:
                                    st.write(summary)
                            except Exception as e:
                                st.warning(f"Could not get agent summary: {str(e)}")
                
                # Handle ChangePointRegimeAgent specifically
                elif agent_name == "changepoint":
                    st.subheader("Change Point Detection Results")
                    
                    # Try to get changepoints and plot if available
                    if hasattr(result, 'get_changepoints') and callable(result.get_changepoints):
                        try:
                            changepoints = result.get_changepoints()
                            if changepoints is not None:
                                st.write(f"Detected {len(changepoints)} change points")
                                
                                # Plot the time series with change points
                                if hasattr(result, 'plot') and callable(result.plot):
                                    fig = result.plot()
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Show change point details if available
                                if hasattr(result, 'get_changepoint_details') and callable(result.get_changepoint_details):
                                    with st.expander("Change Point Details", expanded=False):
                                        details = result.get_changepoint_details()
                                        if details is not None:
                                            st.dataframe(details)
                            else:
                                st.warning("No change points detected")
                        except Exception as e:
                            st.error(f"Error processing change points: {str(e)}")
                    
                    # Show model summary if available
                    if hasattr(result, 'get_summary') and callable(result.get_summary):
                        with st.expander("Model Summary", expanded=False):
                            try:
                                summary = result.get_summary()
                                if isinstance(summary, dict):
                                    st.json(summary)
                                else:
                                    st.write(summary)
                            except Exception as e:
                                st.warning(f"Could not get model summary: {str(e)}")
                
                # Handle agent objects with plot method
                elif hasattr(result, 'plot') and callable(result.plot):
                    try:
                        fig = result.plot()
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
                
                # Handle DataFrames
                elif hasattr(result, 'head') and hasattr(result, 'columns'):
                    st.write("### Data")
                    st.dataframe(result.head())
                    
                    # Show statistics for numerical columns
                    num_cols = result.select_dtypes(include=['float64', 'int64']).columns
                    if len(num_cols) > 0:
                        st.write("### Statistics")
                        st.dataframe(result.describe())
                
                # Handle dictionaries
                elif isinstance(result, dict):
                    # Special handling for UncertaintySynthesisAgent
                    if result.get('agent_type') == 'UncertaintySynthesisAgent' or \
                       (hasattr(result.get('agent', {}), '__class__') and \
                        'UncertaintySynthesisAgent' in str(result['agent'].__class__)):
                        
                        st.subheader("Uncertainty Synthesis Results")
                        
                        # Display the status message
                        status_emoji = "✅" if result.get('status') == 'success' else "⚠️"
                        st.write(f"{status_emoji} {result.get('message', 'Uncertainty synthesis completed')}")
                        
                        # Display a plot if available
                        if 'plot' in result and result['plot'] is not None:
                            st.plotly_chart(result['plot'], use_container_width=True)
                        
                        # Display the transformed data if available
                        if 'transformed_data' in result and result['transformed_data'] is not None:
                            with st.expander("View Data", expanded=False):
                                st.dataframe(result['transformed_data'].head())
                        
                        # Show agent controls if available
                        if 'agent' in result and hasattr(result['agent'], 'get_controls'):
                            with st.expander("Model Controls", expanded=False):
                                controls = result['agent'].get_controls()
                                if controls:
                                    st.write(controls)
                    else:
                        # Default JSON display for other dictionaries
                        st.json(result)
                
                # Handle HTML output
                elif hasattr(result, 'to_html') and callable(result.to_html):
                    st.components.v1.html(result.to_html(), height=400, scrolling=True)
                
                # Handle other types (convert to string)
                else:
                    st.write(str(result))
                
                # Show timestamp if available
                if isinstance(result, dict) and 'timestamp' in result:
                    st.caption(f"Last run: {result['timestamp']}")
                
            except Exception as e:
                st.error(f"Error displaying results from {agent_name}: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language='python')
            
            # Show timestamp if available
            try:
                if isinstance(result, dict):
                    if 'timestamp' in result:
                        st.caption(f"Last run: {result['timestamp']}")
                elif hasattr(result, 'get') and callable(result.get):
                    timestamp = result.get('timestamp')
                    if timestamp is not None:
                        st.caption(f"Last run: {timestamp}")
            except Exception as e:
                # Silently ignore timestamp display errors
                pass

def show_data_page():
    st.header("Input Data")
    
    st.subheader("Current Dataset")
    st.dataframe(st.session_state.df)
    
    st.subheader("Data Statistics")
    st.write(st.session_state.df.describe())
    
    # Plot raw data
    temp_col = get_temp_column(st.session_state.df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.df.index,
        y=st.session_state.df[temp_col],
        name='Temperature',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title="Temperature Time Series",
        xaxis_title="Time",
        yaxis_title=temp_col.replace('_', ' ').title(),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data upload
    st.subheader("Upload New Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            # Check if it has a suitable temperature column
            try:
                get_temp_column(new_df)
                st.session_state.df = new_df
                st.session_state.data_loaded = True
                st.session_state.agents = {}  # Reset agents as data changed
                st.session_state.results = {}  # Clear previous results
                st.success("Data uploaded successfully!")
            except ValueError:
                st.error("Uploaded file must contain a temperature-like column (e.g., 'mean_temp', 'temperature', 'temp')")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()