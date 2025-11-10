import os
import re
from pathlib import Path

# Path to agents directory
agents_dir = Path(__file__).parent / "src" / "agents"

# Process each agent file
for agent_file in agents_dir.glob("agent*.py"):
    print(f"Processing {agent_file.name}...")
    
    # Read the file content
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix relative imports
    new_content = content
    
    # Fix imports from .agent_base
    new_content = re.sub(
        r'from \.agent_base import',
        'from src.agents.agent_base import',
        new_content
    )
    
    # Fix imports from .constants
    new_content = re.sub(
        r'from \.constants import',
        'from src.agents.constants import',
        new_content
    )
    
    # Fix imports from .utils
    new_content = re.sub(
        r'from \.utils import',
        'from src.agents.utils import',
        new_content
    )
    
    # Fix imports from other agent files
    new_content = re.sub(
        r'from \.(agent\d+_\w+) import',
        r'from src.agents.\1 import',
        new_content
    )
    
    # Write the fixed content back
    if new_content != content:
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  Fixed imports in {agent_file.name}")
    else:
        print(f"  No changes needed for {agent_file.name}")

print("\nAll agent files have been processed. Please run the orchestrator again.")
