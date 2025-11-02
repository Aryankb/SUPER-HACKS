import json
import os

# Get the absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the composio.json file
composio_json_path = os.path.join(os.path.dirname(current_dir), 'tools', 'composio.json')

with open(composio_json_path, 'r') as file:
    composio_tools_data = json.load(file)
    print(composio_tools_data)