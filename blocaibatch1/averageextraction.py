import re
import numpy as np

def extract_and_average_matrices(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to capture everything inside the outer brackets [[ ... ]]
    # It accounts for newlines and multi-line rows
    w_matches = re.findall(r'W\s*=\s*(\[\[.*?\]\])', content, re.DOTALL)
    u_matches = re.findall(r'U\s*=\s*(\[\[.*?\]\])', content, re.DOTALL)

    def parse_matrix_string(matrix_str):
        # Clean up the string so it looks like a standard Python list
        # Remove consecutive spaces and replace newlines with commas/spaces
        cleaned = re.sub(re.compile(r'\s+'), ' ', matrix_str)
        cleaned = cleaned.replace('[ ', '[').replace(' ]', ']')
        cleaned = cleaned.replace(' ', ', ')
        # Safely evaluate the string into a nested list
        import ast
        return np.array(ast.literal_eval(cleaned))

    # Parse all extracted string matrices into NumPy arrays
    w_arrays = [parse_matrix_string(m) for m in w_matches]
    u_arrays = [parse_matrix_string(m) for m in u_matches]

    if not w_arrays or not u_arrays:
        print("No matrices found! Check your file path or format.")
        return None, None

    # Calculate the averages across the 0th axis
    avg_W = np.mean(w_arrays, axis=0)
    avg_U = np.mean(u_arrays, axis=0)

    print(f"Successfully processed {len(w_arrays)} agents.")
    print("\n--- Average W Matrix Across File ---")
    print(np.round(avg_W, 4))

    print("\n--- Average U Matrix Across File ---")
    print(np.round(avg_U, 4))

    return avg_W, avg_U


import os

# Get the absolute path of the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
file_matrix_path = os.path.join(script_dir, 'epoch_99.txt')

# Pass the absolute path to your function
avg_w, avg_u = extract_and_average_matrices(file_matrix_path)
print(avg_w,avg_u)