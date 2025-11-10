import os
import glob

def rename_agents():
    agents_dir = os.path.join('src', 'agents')
    
    # Create a mapping of old filenames to new filenames
    file_mapping = {
        '01_': 'agent01_',
        '02_': 'agent02_',
        '03_': 'agent03_',
        '04_': 'agent04_',
        '05_': 'agent05_',
        '06_': 'agent06_',
        '07_': 'agent07_',
        '08_': 'agent08_',
        '09_': 'agent09_',
        '10_': 'agent10_',
        '11_': 'agent11_',
        '12_': 'agent12_',
        '13_': 'agent13_',
        '14_': 'agent14_',
        '15_': 'agent15_',
        '16_': 'agent16_'
    }
    
    # Rename the files
    for old_prefix, new_prefix in file_mapping.items():
        for old_file in glob.glob(os.path.join(agents_dir, f"{old_prefix}*.py")):
            filename = os.path.basename(old_file)
            new_filename = filename.replace(old_prefix, new_prefix, 1)
            new_file = os.path.join(agents_dir, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_agents()
