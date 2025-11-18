import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Set the CSV file path (file must be inside the same folder OR use a URL)
CSV_FILE = r"P:\Optics R&D\Projects\Harmony\Harmony XL Next Gen\Ex vivo China Harmony\acne high after 1.csv"


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_flir_csv(file_path):
    """
    Loads FLIR ResearchIR CSV file into a 2D NumPy array.
    Adjust skiprows if your CSV structure differs.
    """
    try:
        df = pd.read_csv(file_path, header=None, skiprows=10)
        data = df.values.astype(float)
        print(f"Loaded CSV successfully. Shape = {data.shape}")
        return data
    except Exception as e:
        print("Error loading CSV:", e)
        return None


thermal_data = load_flir_csv(CSV_FILE)

if thermal_data is None:
    raise RuntimeError("❌ Cannot run app — data failed to load.")

rows, cols = thermal_data.shape


# -------------------------------------------------------------
# CREATE HEATMAP FIG
# -------------------------------------------------------------
def create_heatmap(data):
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

    fig.update_layout(
        dragmode="select",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest"
    )

    return fig


heatmap_figure = create_heatmap(thermal_data)


# -------------------------------------------------------------
# DASH APP LAYOUT
# -------------------------------------------------------------
app = Dash(__name__)
server = app.server  # For Render deployment

app.layout = html.Div([
    html.H2("FLIR Thermal Image Analyzer", style={'textAlign': 'center'}),

    dcc.Graph(
        id="thermal-heatmap",
        figure=heatmap_figure,
        config={"scrollZoom": True, "displayModeBar": True}
    ),

    html.Div(id="mean-temp-output", style={
        'marginTop': '20px',
        'padding': '10px',
        'border': '1px solid #ccc',
        'fontSize': '1.2em',
        'textAlign': 'center'
    }),

    html.Div(f"Image size: {cols} × {rows} pixels", style={'textAlign': 'center', 'marginTop': '10px'})
])


# -------------------------------------------------------------
# CALLBACK — CALCULATE MEAN / MIN / MAX
# -------------------------------------------------------------
@app.callback(
    Output("mean-temp-output", "children"),
    Input("thermal-heatmap", "selectedData")
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "👉 Select a region on the heatmap."

    try:
        # -----------------------------------------------
        # Case 1: Box selection gives "range"
        # -----------------------------------------------
        if "range" in selectedData:
            xr = selectedData["range"]["x"]
            yr = selectedData["range"]["y"]
            x_min, x_max = int(xr[0]), int(xr[1])
            y_min, y_max = int(yr[0]), int(yr[1])

        # -----------------------------------------------
        # Case 2: Fallback to "points" list
        # -----------------------------------------------
        elif "points" in selectedData and selectedData["points"]:
            xs = [p["x"] for p in selectedData["points"]]
            ys = [p["y"] for p in selectedData["points"]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

        else:
            return "⚠️ Nothing selected — try drawing a box."

        # Clamp to valid boundaries
        x_min = max(0, int(np.floor(x_min)))
        x_max = min(cols - 1, int(np.ceil(x_max)))
        y_min = max(0, int(np.floor(y_min)))
        y_max = min(rows - 1, int(np.ceil(y_max)))

        region = thermal_data[y_min:y_max + 1, x_min:x_max + 1]

        mean_val = float(np.mean(region))
        min_val = float(np.min(region))
        max_val = float(np.max(region))
        pixels = region.size

        # Print to server logs
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
# MAIN ENTRY (important for deployment)
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
