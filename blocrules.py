import tkinter as tk

# Your input dictionary
data_dict = {
    ((1, -1), (1, 0)): [[-1, 1], [1, 0]],
    ((1, -1), (1, 1)): [[-1, 1], [1, 1]],
    ((1, -1), (0, 0)): [[-1, -1], [0, 0]],
    ((1, -1), (0, 1)): [[-1, 1], [1, 0]],
    ((1, 1), (1, 0)): [[1, 1], [1, 1]],
    ((1, 1), (1, 1)): [[1, 1], [1, 0]],
    ((1, 1), (0, 0)): [[-1, 1], [0, 1]],
    ((1, 1), (0, 1)): [[-1, 1], [1, 1]],
    ((1, 0), (1, 0)): [[1, 0], [0, 1]],
    ((1, 0), (0, 1)): [[-1, -1], [1, 1]],
    ((1, 0), (1, 1)): [[1, 1], [1, 0]],
    ((1, 0), (0, 0)): [[0, 0], [1, 1]],
    ((1, -1), (0, -1)): [[-1, 1], [0, 1]],
    ((1, -1), (1, -1)): [[-1, 1], [1, 1]],
    ((1, 1, 0), (1, 1, 1)): [[1, 1, 1], [1, 1, 0]],
    ((1, 1, -1), (1, 1, 1)): [[1, 1, 1], [1, 1, 1]]
}

descriptions = [
    "Easy to travel on home side", "Easy to travel on home side", 
    "Don't get swallowed up!", "Earthshaking", "Absorption", 
    "De-absorption", "Clockwise", "Clockwise", "Home turf? No turf!", 
    "Your crater", "Clockwise", "Kamikaze", "Ground gain", 
    "Ground gain", "Swaptory", "Factory"
]

def get_circle_color(val):
    if val == 1: return "black"
    if val == 0: return "white"
    return ""  # -1 is empty

def get_square_color(val):
    if val == 1: return "black"
    if val == 0: return "white"
    if val == -1: return "gray"
    return "white"

def draw_state_block(canvas, x, y, top_row, bottom_row):
    for i, val in enumerate(top_row):
        x0 = x + i * (CELL_SIZE + SPACING)
        y0 = y
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        fill_color = get_circle_color(val)
        if fill_color:
            canvas.create_oval(x0, y0, x1, y1, fill=fill_color, outline="black")
        else:
            canvas.create_oval(x0, y0, x1, y1, fill="", outline="black", dash=(3, 3))

    for i, val in enumerate(bottom_row):
        x0 = x + i * (CELL_SIZE + SPACING)
        y0 = y + CELL_SIZE + SPACING
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        fill_color = get_square_color(val)
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="black")
        
    max_len = max(len(top_row), len(bottom_row))
    return max_len * (CELL_SIZE + SPACING) - SPACING

# Layout Configuration
CELL_SIZE = 20
SPACING = 6
ARROW_LENGTH = 40
ROW_PADDING = 50
COLUMNS_PER_ROW = 2 

root = tk.Tk()
root.title("State Transition Visualizer")
root.geometry("950x700")  # Constrain window dimensions to fit your VNC 1024x768 layout

# --- Scrollable Canvas Setup ---
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Create a canvas with a large scroll region height (e.g., 1200 pixels) to handle all rows
canvas = tk.Canvas(frame, bg="#f5f5f5", scrollregion=(0, 0, 950, 1200))

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Mousewheel scrolling support
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

# --- Drawing Logic ---
current_row = 0
current_col = 0
start_x = 50
start_y = 50

for index, ((key, value), desc) in enumerate(zip(data_dict.items(), descriptions)):
    if current_col >= COLUMNS_PER_ROW:
        current_col = 0
        current_row += 1
        start_x = 50
        start_y = 50 + current_row * (CELL_SIZE * 2 + SPACING * 2 + ROW_PADDING)

    # 1. Draw Key State
    key_width = draw_state_block(canvas, start_x, start_y + 20, key[0], key[1])
    
    # 2. Draw Arrow
    arrow_start_x = start_x + key_width + 15
    arrow_end_x = arrow_start_x + ARROW_LENGTH
    arrow_y = (start_y + 20) + CELL_SIZE + (SPACING / 2)
    canvas.create_line(arrow_start_x, arrow_y, arrow_end_x, arrow_y, arrow=tk.LAST, width=2, fill="#555555")
    
    # 3. Draw Value State
    value_start_x = arrow_end_x + 15
    value_width = draw_state_block(canvas, value_start_x, start_y + 20, value[0], value[1])
    
    # 4. Draw Centered Text Label
    total_unit_width = key_width + 30 + ARROW_LENGTH + value_width
    text_center_x = start_x + (total_unit_width / 2)
    canvas.create_text(text_center_x, start_y, text=desc, font=("Arial", 11, "bold"), fill="#2c3e50", anchor="n")
    
    start_x += total_unit_width + 100 
    current_col += 1

root.mainloop()