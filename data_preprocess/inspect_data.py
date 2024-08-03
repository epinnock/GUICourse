import os
import json
import glob

def count_sft_instruction_entries():
    file_counts = {}
    file_pattern = "./data/*_sft_instructions.json"
    
    for file_path in glob.glob(file_pattern):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    file_counts[filename] = len(data)
                elif isinstance(data, dict):
                    file_counts[filename] = len(data.keys())
                else:
                    file_counts[filename] = 1
        except json.JSONDecodeError:
            print(f"Error: Unable to parse {filename}")
        except IOError:
            print(f"Error: Unable to read {filename}")
    
    return file_counts

# Example usage
if __name__ == "__main__":
    results = count_sft_instruction_entries()
    
    for filename, count in results.items():
        print(f"{filename}: {count} entries")

    total_entries = sum(results.values())
    print(f"\nTotal entries across all files: {total_entries}")
