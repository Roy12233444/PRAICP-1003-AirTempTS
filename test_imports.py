import sys
import os
from pathlib import Path

# Print Python version and paths
print(f"Python {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

# Add project root and src to path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("\nAttempting to import agents...")
try:
    from agents.agent01_resonant_decomposition_agent import ResonantDecompositionAgent
    print("Successfully imported ResonantDecompositionAgent")
except Exception as e:
    print(f"Error importing agent: {e}")
    print("\nTrying to list agent files:")
    agent_files = list((src_path / "agents").glob("agent*.py"))
    print(f"Found {len(agent_files)} agent files:")
    for f in agent_files:
        print(f"  - {f.name}")
