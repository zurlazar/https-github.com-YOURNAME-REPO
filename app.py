import pandas as pd
import numpy as np
import plotly.express as px
import dash
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# -------------------------------------------------------------
# CSV LISTING
# -------------------------------------------------------------
def list_csv_files(folder="."):
    """Lists all CSV files in the current folder."""
    return [f for f in os.listdir(folder) if f.lower().endswith(".csv")]

CSV_FILES = list_csv_files(".")
if not CSV_FILES:
    # Use a dummy entry to prevent startup failure if no files are present
    print("❌ No CSV files found in folder. Using default empty state.")
    CSV_FILES = ["default.csv"]

# Global variables to store the currently loaded data
thermal_data = None
rows = 0
cols = 0

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_flir_csv(file_path):
    """
    Loads FLIR ResearchIR CSV file into a 2D NumPy array.
    Updates the global thermal_data and shape variables.
    """
    # Define thermal_data, rows, and cols as global inside the function
    global thermal_data, rows, cols
    
    if file_path == "default.csv":
        # Create a tiny dummy array if no real CSVs were found
        thermal_data = np.zeros((10, 10))
        rows, cols = 10, 10
        print("Using empty data state (10x10).")
        return thermal_data
        
    try:
        df = pd.read_csv(file_path, header=None, skiprows=10)
        data = df.values.astype(float)
        print(f"Loaded CSV: {file_path}  Shape={data.shape}")
        
        # Update global variables after successful load
        thermal_data = data
        rows, cols = thermal_data.shape
        
        return data
    except Exception as e:
        print("Error loading CSV:", e)
        # Reset globals on failure
        thermal_data = None
        rows, cols = 0, 0
        return None

# Load first CSV initially
if not load_flir_csv(CSV_FILES[0]) is not None and CSV_FILES[0] != "default.csv":
    # Only raise error if a real CSV failed to load and it's not the dummy state
    if CSV_FILES[0] != "default.csv":
        raise RuntimeError("❌ Cannot run app — initial data failed to load.")
    
# -------------------------------------------------------------
# CREATE FIGURE
# -------------------------------------------------------------
def create_heatmap(data, min_temp=None, max_temp=None): 
    """
    Creates a heatmap with optional user-defined color range.
    """
    if data is None or data.size == 0:
        return {} # Return empty figure if no data
        
    # Use the user-provided min/max, otherwise calculate from data
    min_val = min_temp if min_temp is not None else np.nanmin(data)
    max_val = max_temp if max_temp is not None else np.nanmax(data)

    fig = px.imshow(
        data,
        color_continuous_scale="Inferno",
        range_color=[min_val, max_val], # Use the determined min/max
        aspect="equal",
        title="Thermal Heatmap — Select an Area",
        labels={'x': 'X Pixel', 'y': 'Y Pixel', 'color': 'Temperature'}
    )
    fig.update_layout(dragmode="select", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# -------------------------------------------------------------
# DASH APP LAYOUT
# -------------------------------------------------------------
# Define app and server globally for deployment services (e.g., Render)
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    # --- START SIGNATURE BLOCK ---
    html.Div([
        html.H2("Thermal Image Analyzer", style={'textAlign': 'left', 'display': 'inline-block', 'marginRight': '20px'}),
        html.Div(
            "Made by Zur Lazar Nov, 19th 2025",
            style={
                'fontFamily': 'Georgia, serif',
                'fontSize': '1.0em',
                'color': '#444',
                'fontStyle': 'italic',
                'fontWeight': 'bold',
                'float': 'right',
                'paddingTop': '15px'
            }
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'baseline', 'width': '80%', 'margin': 'auto'}),
    # --- END SIGNATURE BLOCK ---

    # ---------------------------- CONTROLS ROW ----------------------------
    html.Div([
        # CSV Selector
        html.Div([
            html.Label("Select CSV File:"),
            dcc.Dropdown(
                id="csv-selector",
                options=[{"label": f, "value": f} for f in CSV_FILES],
                value=CSV_FILES[0],
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        # Minimum Temperature Input
        html.Div([
            html.Label("Min Temp Scale:"),
            dcc.Input(
                id="min-temp-input",
                type="number",
                placeholder="Auto Min",
                style={'width': '100%'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        # Maximum Temperature Input
        html.Div([
            html.Label("Max Temp Scale:"),
            dcc.Input(
                id="max-temp-input",
                type="number",
                placeholder="Auto Max",
                style={'width': '100%'}
            )
        ], style={'width': '15%', 'display': 'inline-block'}),
        
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    dcc.Graph(
        id="thermal-heatmap",
        # Initial figure uses auto min/max (None, None)
        figure=create_heatmap(thermal_data, None, None), 
        config={"scrollZoom": True, "displayModeBar": True}
    ),

    html.Div(id="mean-temp-output", style={
        'marginTop': '20px',
        'padding': '10px',
        'border': '1px solid #ccc',
        'fontSize': '1.2em',
        'textAlign': 'center'
    }),

    html.Div(f"Image size: {cols} × {rows} pixels", id="image-size-text", style={
        'textAlign': 'center',
        'marginTop': '10px'
    })
])


# -------------------------------------------------------------
# CALLBACK 1: CHANGE CSV FILE & UPDATE HEATMAP
# -------------------------------------------------------------
@app.callback(
    Output("thermal-heatmap", "figure", allow_duplicate=True),
    Output("image-size-text", "children"),
    Input("csv-selector", "value"),
    prevent_initial_call=True
)
def update_file(selected_csv):
    """Loads a new CSV, updates global data, and redraws the heatmap (with auto scale)."""
    load_flir_csv(selected_csv)
    
    # Redraw heatmap with default (auto) min/max scale
    fig = create_heatmap(thermal_data, None, None) 
    return fig, f"Image size: {cols} × {rows} pixels"


# -------------------------------------------------------------
# CALLBACK 2: UPDATE HEATMAP FIGURE BASED ON SCALE INPUTS
# -------------------------------------------------------------
@app.callback(
    Output("thermal-heatmap", "figure", allow_duplicate=True),
    Input("min-temp-input", "value"),
    Input("max-temp-input", "value"),
    Input("image-size-text", "children"), 
    prevent_initial_call="callback-triggered"
)
def update_heatmap_scale(min_val_str, max_val_str, trigger_csv_load):
    """Redraws the heatmap using the selected or default color scale."""
    
    # 1. Convert inputs from string/None to float/None
    min_temp = float(min_val_str) if min_val_str is not None and min_val_str != '' else None
    max_temp = float(max_val_str) if max_val_str is not None and max_val_str != '' else None
    
    # 2. Re-create the figure using the global thermal_data and the user inputs
    fig = create_heatmap(thermal_data, min_temp, max_temp)
    return fig


# -------------------------------------------------------------
# CALLBACK 3: CALCULATE MEAN/STD/MIN/MAX (REGION SELECT)
# -------------------------------------------------------------
@app.callback(
    Output("mean-temp-output", "children"),
    Input("thermal-heatmap", "selectedData")
)
def display_selected_data(selectedData):
    """Calculates Mean, Std Dev, Min, and Max for the selected region."""
    if selectedData is None:
        return "👉 Select a region on the heatmap."

    try:
        global thermal_data, rows, cols
        
        if thermal_data is None or rows == 0 or cols == 0:
             return "⚠️ Please load a valid CSV file first."

        # Box range extraction (Handles box select, lasso, etc.)
        if "range" in selectedData:
            xr = selectedData["range"]["x"]
            yr = selectedData["range"]["y"]
            x_min, x_max = int(xr[0]), int(xr[1])
            y_min, y_max = int(yr[0]), int(yr[1])
        # Points fallback
        elif "points" in selectedData and selectedData["points"]:
            xs = [p["x"] for p in selectedData["points"]]
            ys = [p["y"] for p in selectedData["points"]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        else:
            return "⚠️ Nothing selected — try box select."

        # Clamp boundaries to valid pixel indices
        x_min = max(0, int(np.floor(x_min)))
        x_max = min(cols - 1, int(np.ceil(x_max)))
        y_min = max(0, int(np.floor(y_min)))
        y_max = min(rows - 1, int(np.ceil(y_max)))

        region = thermal_data[y_min:y_max + 1, x_min:x_max + 1]
        
        if region.size == 0:
             return "⚠️ Selected area is zero size."

        mean_val = float(np.nanmean(region))
        std_val = float(np.nanstd(region)) 
        min_val = float(np.nanmin(region))
        max_val = float(np.nanmax(region))

        print(f"[REGION] mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")

        return html.Div([
            html.P(f"Mean: {mean_val:.3f}", style={'fontSize': '1.5em'}),
            html.P(f"Std Dev: {std_val:.3f}", style={'color': 'gray'}), 
            html.P(f"Min: {min_val:.3f}", style={'color': 'blue'}),
            html.P(f"Max: {max_val:.3f}", style={'color': 'red'}),
        ])

    except Exception as e:
        print("Error calculating region statistics:", e)
        return "❌ Error calculating region statistics."

# -------------------------------------------------------------
# MAIN ENTRY (Removed for deployment)
# -------------------------------------------------------------
# The deployment service (like Render) will look for the global 'server' variable
# and run it. The manual app.run() call is not needed or desired here.
