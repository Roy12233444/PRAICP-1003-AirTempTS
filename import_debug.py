import os
import sys
from pathlib import Path

print("=== Debugging Imports ===\n")

# Print current working directory
print(f"Current working directory: {os.getcwd()}")
print("\nFiles in current directory:")
for f in os.listdir():
    print(f"- {f}")

# Check src directory
src_path = str(Path(__file__).parent / 'src')
print(f"\nLooking for src directory at: {src_path}")

if os.path.exists(src_path):
    print("\nContents of src directory:")
    for f in os.listdir(src_path):
        print(f"- {f}")
    
    # Check agents directory
    agents_path = os.path.join(src_path, 'agents')
    if os.path.exists(agents_path):
        print("\nContents of src/agents directory:")
        for f in os.listdir(agents_path):
            print(f"- {f}")
        
        # Check if agent files exist
        agent_files = [
            'agent01_resonant_decomposition_agent.py',
            'agent02_wavelet_transient_agent.py',
            'agent_registry.py',
            'agent_base.py'
        ]
        
        print("\nChecking for required agent files:")
        for agent_file in agent_files:
            full_path = os.path.join(agents_path, agent_file)
            print(f"- {agent_file}: {'Found' if os.path.exists(full_path) else 'Missing'}")
        
        # Try to import agent_registry
        print("\nAttempting to import agent_registry...")
        try:
            # Add src to path if not already there
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            from agents.agent_registry import REGISTRY
            print("SUCCESS: Imported REGISTRY from agent_registry")
            print(f"Available agents: {list(REGISTRY.keys())}")
            
            # Try to create an agent instance
            print("\nAttempting to create agent instance...")
            try:
                from agents.agent_registry import get_agent
                agent = get_agent("resonant")
                print(f"SUCCESS: Created agent instance: {agent}")
            except Exception as e:
                print(f"ERROR creating agent: {e}")
                
        except Exception as e:
            print(f"ERROR importing agent_registry: {e}")
            print("\nCurrent sys.path:")
            for p in sys.path:
                print(f"- {p}")
    else:
        print(f"\nERROR: Agents directory not found at {agents_path}")
else:
    print(f"\nERROR: src directory not found at {src_path}")

print("\n=== End of Debugging ===")
