from pathlib import Path
import re

# Path to agents directory
agents_dir = Path(__file__).parent / "src" / "agents"

# Process each agent file
for agent_file in agents_dir.glob("agent*.py"):
    print(f"Processing {agent_file.name}...")
    
    # Read the file content
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix relative imports
    new_content = re.sub(
        r'from \.([a-zA-Z0-9_]+) import',
        r'from agents.\1 import',
        content
    )
    
    # Write the fixed content back
    if new_content != content:
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  Fixed imports in {agent_file.name}")
    else:
        print(f"  No changes needed for {agent_file.name}")

print("\nAll agent files have been processed. Please run the orchestrator again.")
