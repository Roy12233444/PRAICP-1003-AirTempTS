import sys
import os
from pathlib import Path

# Add the project root and src to Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import and run the orchestrator
from agents.autogen_orchestrator import main

if __name__ == "__main__":
    print(f"Running with project root: {project_root}")
    print(f"Python path: {sys.path}")
    
    data_file = project_root / "data" / "surface-air-temperature-monthly-mean.csv"
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
        
    main(
        project_root=str(project_root),
        run_agents=True,
        run_csv=str(data_file)
    )
