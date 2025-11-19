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
    return [f for f in os.listdir(folder) if f.lower().endswith(".csv")]

CSV_FILES = list_csv_files(".")
if not CSV_FILES:
    raise RuntimeError("❌ No CSV files found in folder.")

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
    """
    try:
        df = pd.read_csv(file_path, header=None, skiprows=10)
        data = df.values.astype(float)
        print(f"Loaded CSV: {file_path}  Shape={data.shape}")
        
        # Update global variables after successful load
        global thermal_data, rows, cols
        thermal_data = data
        rows, cols = thermal_data.shape
        
        return data
    except Exception as e:
        print("Error loading CSV:", e)
        return None

# Load first CSV initially and handle error
if not load_flir_csv(CSV_FILES[0]) is not None:
    raise RuntimeError("❌ Cannot run app — initial data failed to load.")

# -------------------------------------------------------------
# CREATE FIGURE
# -------------------------------------------------------------
def create_heatmap(data, min_temp=None, max_temp=None):
    """
    Creates a heatmap with optional user-defined color range.
    """
    # Use the data's min/max if not provided by user
    min_val = min_temp if min_temp is not None else np.nanmin(data)
    max_val = max_temp if max_temp is not None else np.nanmax(data)

    fig = px.imshow(
        data,
        color_continuous_scale="Inferno",
        range_color=[min_val, max_val], # Use the provided min/max
        aspect="equal",
        title="FLIR Thermal Heatmap — Select an Area",
        labels={'x': 'X Pixel', 'y': 'Y Pixel', 'color': 'Temperature'}
    )
    fig.update_layout(dragmode="select", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# -------------------------------------------------------------
# DASH APP LAYOUT
# -------------------------------------------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("FLIR Thermal Image Analyzer", style={'textAlign': 'center'}),

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
            html.Label("Min Temperature Scale:"),
            dcc.Input(
                id="min-temp-input",
                type="number",
                placeholder="Auto Min",
                style={'width': '100%'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        # Maximum Temperature Input
        html.Div([
            html.Label("Max Temperature Scale:"),
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
        # Initial figure uses the default auto-scale
        figure=create_heatmap(thermal_data),
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
# CALLBACK 1: CHANGE CSV FILE & UPDATE FIGURE GLOBALS
# -------------------------------------------------------------
# This callback updates the global data object whenever a new CSV is selected.
@app.callback(
    Output("image-size-text", "children"), # Just an output to trigger the data load
    Input("csv-selector", "value"),
)
def update_file(selected_csv):
    load_flir_csv(selected_csv)
    # The figure update is handled by the second callback below
    return f"Image size: {cols} × {rows} pixels"


# -------------------------------------------------------------
# CALLBACK 2: UPDATE HEATMAP FIGURE BASED ON SCALE INPUTS
# -------------------------------------------------------------
# This callback redraws the figure whenever the CSV is loaded OR the min/max inputs change.
@app.callback(
    Output("thermal-heatmap", "figure"),
    # The figure needs to be updated when the CSV changes (triggering the first input) 
    # OR when the min/max inputs change.
    Input("image-size-text", "children"), # Trigger when the CSV changes (Output from Callback 1)
    Input("min-temp-input", "value"),
    Input("max-temp-input", "value"),
    # State is needed to pass the current data without triggering an update
)
def update_heatmap_scale(trigger_csv_load, min_val_str, max_val_str):
    
    # 1. Convert inputs from string/None to float/None
    min_temp = float(min_val_str) if min_val_str is not None else None
    max_temp = float(max_val_str) if max_val_str is not None else None
    
    # 2. Re-create the figure using the global thermal_data and the user inputs
    fig = create_heatmap(thermal_data, min_temp, max_temp)
    return fig


# -------------------------------------------------------------
# CALLBACK — CALCULATE MEAN/MIN/MAX (UNCHANGED)
# -------------------------------------------------------------
@app.callback(
    Output("mean-temp-output", "children"),
    Input("thermal-heatmap", "selectedData")
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "👉 Select a region on the heatmap."

    try:
        # Use the global data which is guaranteed to be loaded
        global thermal_data, rows, cols

        # Box range
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

        # Clamp
        x_min = max(0, int(np.floor(x_min)))
        x_max = min(cols - 1, int(np.ceil(x_max)))
        y_min = max(0, int(np.floor(y_min)))
        y_max = min(rows - 1, int(np.ceil(y_max)))

        region = thermal_data[y_min:y_max + 1, x_min:x_max + 1]

        mean_val = float(np.mean(region))
        min_val = float(np.min(region))
        max_val = float(np.max(region))
        pixels = region.size

        print(f"[REGION] mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}, pixels={pixels}")

        return html.Div([
            html.P(f"Mean: {mean_val:.3f}", style={'fontSize': '1.5em'}),
            html.P(f"Min: {min_val:.3f}", style={'color': 'blue'}),
            html.P(f"Max: {max_val:.3f}", style={'color': 'red'}),
            html.P(f"Pixels: {pixels}")
        ])

    except Exception as e:
        print("Error:", e)
        return "❌ Error calculating region statistics."


# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
