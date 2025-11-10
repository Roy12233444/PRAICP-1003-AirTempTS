import sys
import os
from pathlib import Path

# Add the project root and src to Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

# Add to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Print debug info
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Import the orchestrator
try:
    from agents.autogen_orchestrator import main
    
    # Run with test data
    data_file = project_root / "data" / "surface-air-temperature-monthly-mean.csv"
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
        
    print(f"\nRunning orchestrator with data: {data_file}")
    main(
        project_root=str(project_root),
        run_agents=True,
        run_csv=str(data_file)
    )
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
