import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the orchestrator
from src.agents.autogen_orchestrator import main

if __name__ == "__main__":
    main(
        project_root=str(project_root),
        run_agents=True,
        run_csv=str(project_root / "data" / "surface-air-temperature-monthly-mean.csv")
    )
