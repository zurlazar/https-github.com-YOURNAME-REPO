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
    raise RuntimeError("❌ No CSV files found in folder.")

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
        print(f"Loaded CSV: {file_path}  Shape={data.shape}")
        return data
    except Exception as e:
        print("Error loading CSV:", e)
        return None

# load first CSV initially
thermal_data = load_flir_csv(CSV_FILES[0])
# Ensure global variables are set (important for the callbacks)
global rows, cols
if thermal_data is not None:
    rows, cols = thermal_data.shape
else:
    # If loading failed, use placeholders to prevent errors
    rows, cols = 0, 0 
    
# -------------------------------------------------------------
# CREATE FIGURE
# -------------------------------------------------------------
def create_heatmap(data):
    """
    Creates a heatmap with auto min/max temperature scaling.
    """
    if data is None or data.size == 0:
        return {} # Return empty figure if no data
        
    MIN_TEMP = np.nanmin(data)
    MAX_TEMP = np.nanmax(data)

    fig = px.imshow(
        data,
        color_continuous_scale="Inferno",
        range_color=[MIN_TEMP, MAX_TEMP],
        aspect="equal",
        title="FLIR Thermal Heatmap — Select an Area",
        labels={'x': 'X Pixel', 'y': 'Y Pixel', 'color': 'Temperature'}
    )
    # dragmode="select" enables the standard box selection tool
    fig.update_layout(dragmode="select", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# -------------------------------------------------------------
# DASH APP
# -------------------------------------------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("FLIR Thermal Image Analyzer", style={'textAlign': 'center'}),

    # ---------------------------- CSV SELECTOR ----------------------------
    html.Div([
        html.Label("Select CSV File:"),
        dcc.Dropdown(
            id="csv-selector",
            options=[{"label": f, "value": f} for f in CSV_FILES],
            value=CSV_FILES[0],
            clearable=False
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    dcc.Graph(
        id="thermal-heatmap",
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
# CALLBACK — CHANGE CSV FILE & UPDATE HEATMAP
# -------------------------------------------------------------
@app.callback(
    Output("thermal-heatmap", "figure"),
    Output("image-size-text", "children"),
    Input("csv-selector", "value"),
)
def update_file(selected_csv):
    global thermal_data, rows, cols

    thermal_data = load_flir_csv(selected_csv)
    # Check for successful load before getting shape
    if thermal_data is not None:
        rows, cols = thermal_data.shape
    else:
        rows, cols = 0, 0

    fig = create_heatmap(thermal_data)
    return fig, f"Image size: {cols} × {rows} pixels"


# -------------------------------------------------------------
# CALLBACK — CALCULATE MEAN/STD/MIN/MAX (MODIFIED)
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
        
        # Ensure we have data before slicing
        if thermal_data is None:
             return "⚠️ Please load a valid CSV file first."

        # Box range extraction (Handles box select, lasso, etc.)
        if "range" in selectedData:
            xr = selectedData["range"]["x"]
            yr = selectedData["range"]["y"]
            x_min, x_max = int(xr[0]), int(xr[1])
            y_min, y_max = int(yr[0]), int(yr[1])
        # Points fallback (Handles point/individual selections)
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

        # Slice the data for the selected region (y_max + 1 to include the last row/column)
        region = thermal_data[y_min:y_max + 1, x_min:x_max + 1]
        
        if region.size == 0:
             return "⚠️ Selected area is zero size."

        # --- MODIFIED: Calculate Std Dev and remove Pixels ---
        mean_val = float(np.nanmean(region))
        std_val = float(np.nanstd(region)) # Calculate Standard Deviation
        min_val = float(np.nanmin(region))
        max_val = float(np.nanmax(region))
        # pixels = region.size # Removed pixel count calculation
        # -----------------------------------------------------

        print(f"[REGION] mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")

        # --- MODIFIED OUTPUT FORMAT ---
        return html.Div([
            html.P(f"Mean: {mean_val:.3f}", style={'fontSize': '1.5em'}),
            html.P(f"Std Dev: {std_val:.3f}", style={'color': 'gray'}), # New line for Std Dev
            html.P(f"Min: {min_val:.3f}", style={'color': 'blue'}),
            html.P(f"Max: {max_val:.3f}", style={'color': 'red'}),
            # Removed: html.P(f"Pixels: {pixels}") 
        ])
        # ------------------------------

    except Exception as e:
        print("Error calculating region statistics:", e)
        return "❌ Error calculating region statistics."


# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
if __name__ == "__main__":
    # NOTE: Using 0.0.0.0 is correct for Render deployment but may cause
    # issues in some local IDE/Jupyter environments. Use 127.0.0.1 for local debugging.
    # Since you are preparing for GitHub/Render, 0.0.0.0 is the correct final deployment host.
    app.run_server(host="0.0.0.0", port=8050, debug=True)
