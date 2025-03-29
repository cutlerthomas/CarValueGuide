import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Load CSV data
df = pd.read_csv('final_vehicle_data.csv')

# Define which columns are categorical and which are numeric
categorical_columns = [
    'Make', 'Model', 'Engine Fuel Type', 'Engine Cylinders',
    'Transmission Type', 'Driven_Wheels', 'Number of Doors',
    'Market Category', 'Vehicle Size', 'Vehicle Style', 'cluster', 'meta_cluster'
]
numeric_columns = ['Engine HP', 'highway MPG', 'city mpg', 'MSRP', 'value_score', 'meta_value_score']

# Create filter components for categorical columns
categorical_filters = []
for col in categorical_columns:
    options = [{'label': str(val), 'value': val} 
               for val in sorted(df[col].dropna().unique())]
    categorical_filters.append(
        html.Div([
            html.Label(f"Select {col}:"),
            dcc.Dropdown(
                id=f"{col.replace(' ', '_').lower()}-dropdown",
                options=options,
                multi=True,
                placeholder=f"Filter by {col}..."
            )
        ], style={'marginBottom': '15px'})
    )

# Create filter components for numeric columns
numeric_filters = []
for col in numeric_columns:
    min_val = df[col].min()
    max_val = df[col].max()
    # A basic step calculation; adjust this as needed
    step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 1
    numeric_filters.append(
        html.Div([
            html.Label(f"Select {col} Range:"),
            dcc.RangeSlider(
                id=f"{col.replace(' ', '_').lower()}-slider",
                min=min_val,
                max=max_val,
                step=step,
                value=[min_val, max_val],
                marks={
                    int(min_val): str(int(min_val)),
                    int(max_val): str(int(max_val))
                }
            )
        ], style={'marginBottom': '25px'})
    )

all_filters = categorical_filters + numeric_filters

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the app layout
app.layout = html.Div([
    # Button to open the offcanvas sidebar
    dbc.Button(
        "Filters",
        id="open-offcanvas",
        n_clicks=0,
        style={"position": "fixed", "top": "10px", "left": "10px", "zIndex": 1000}
    ),
    # Offcanvas sidebar with all filter components
    dbc.Offcanvas(
        html.Div(all_filters, style={'padding': '10px'}),
        id="offcanvas",
        title="Filter Options",
        is_open=False,
        placement="start",
        backdrop=True
    ),
    # Main content area: the graph
     html.Div([
        dcc.Graph(
            id='pca-scatter',
            style={'height': '50vh', 'width': '100%'}  # Graph takes up the top half of the viewport
        ),
        html.Div(
            id='car-details',
            style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}
        )
    ], style={'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '70px'})
])

# Toggle sidebar open/close
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    State("offcanvas", "is_open")
)

def toggle_offcanvas(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Create a callback that takes the values from each filter
@app.callback(
    Output('pca-scatter', 'figure'),
    [Input(f"{col.replace(' ', '_').lower()}-dropdown", 'value') for col in categorical_columns] +
    [Input(f"{col.replace(' ', '_').lower()}-slider", 'value') for col in numeric_columns]
)
def update_graph(*args):
    # Separate the inputs: first n are categorical, then numeric inputs follow
    num_cat = len(categorical_columns)
    cat_values = args[:num_cat]
    num_values = args[num_cat:]
    
    filtered_df = df.copy()
    
    # Apply filters for categorical columns
    for i, col in enumerate(categorical_columns):
        selected = cat_values[i]
        if selected and len(selected) > 0:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    
    # Apply filters for numeric columns
    for i, col in enumerate(numeric_columns):
        slider_range = num_values[i]
        if slider_range:
            filtered_df = filtered_df[(filtered_df[col] >= slider_range[0]) & 
                                      (filtered_df[col] <= slider_range[1])]
            
    filtered_df = filtered_df.reset_index()
    
    # Create a 3D scatter plot using PCA coordinates (assuming they exist)
    if all(c in filtered_df.columns for c in ['P1', 'P2', 'P3']):
        fig = px.scatter_3d(
            filtered_df,
            x='P1', y='P2', z='P3',
            color='meta_cluster',  # You can change the color mapping as desired
            hover_data=['Make', 'Model', 'Year', 'MSRP', 'value_score'],
            custom_data=['index'],
            title="3D Scatter Plot of Vehicle Clusters"
        )
        fig.update_layout(transition_duration=500)
    else:
        # Fallback: if PCA coordinates are not available, plot two numeric columns
        fig = px.scatter(
            filtered_df,
            x=numeric_columns[0], y=numeric_columns[1],
            title="Scatter Plot (PCA coordinates not available)"
        )
    
    return fig

# Callback to display details of a clicked datapoint below the graph
@app.callback(
    Output('car-details', 'children'),
    Input('pca-scatter', 'clickData')
)

def display_car_details(clickData):
    if clickData is None:
        return "Click on a datapoint to see details for that vehicle."
    
    # Extract the custom data (the index) from the clicked point
    point_index = clickData['points'][0]['customdata'][0]
    # Retrieve the corresponding row from the original DataFrame
    car_data = df.loc[point_index]
    
    # Build a table to display all of the car's data
    rows = []
    for col, val in car_data.items():
        rows.append(html.Tr([html.Td(html.B(col)), html.Td(str(val))]))
    
    table = html.Table(rows, style={'width': '100%', 'border': '1px solid #ccc', 'borderCollapse': 'collapse'})
    return table

if __name__ == '__main__':
    app.run(debug=True)
