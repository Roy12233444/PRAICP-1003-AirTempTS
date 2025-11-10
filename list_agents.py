from pathlib import Path

# List all agent files
agents_dir = Path(__file__).parent / "src" / "agents"
agent_files = list(agents_dir.glob("agent*.py"))

print(f"Found {len(agent_files)} agent files:")
for f in agent_files:
    print(f"- {f.name}")
    
    # Read first few lines to check imports
    try:
        with open(f, 'r', encoding='utf-8') as file:
            print("  First 5 lines:")
            for i, line in enumerate(file):
                if i >= 5:
                    break
                print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"  Error reading file: {e}")
