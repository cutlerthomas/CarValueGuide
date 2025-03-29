import dash
from dash import dcc, html, Input, Output
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

# Initialize the Dash app
app = dash.Dash(__name__)

# Create filter components for categorical columns
categorical_filters = []
for col in categorical_columns:
    options = [{'label': str(val), 'value': val} 
               for val in sorted(df[col].dropna().unique())]
    dropdown = html.Div([
        html.Label(f"Select {col}:"),
        dcc.Dropdown(
            id=f"{col.replace(' ', '_').lower()}-dropdown",
            options=options,
            multi=True,
            placeholder=f"Filter by {col}..."
        )
    ], style={'width': '40%', 'margin': '10px', 'display': 'inline-block'})
    categorical_filters.append(dropdown)

# Create filter components for numeric columns
numeric_filters = []
for col in numeric_columns:
    min_val = df[col].min()
    max_val = df[col].max()
    # A basic step calculation; adjust this as needed
    step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 1
    slider = html.Div([
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
    ], style={'width': '40%', 'margin': '10px', 'display': 'inline-block'})
    numeric_filters.append(slider)

# Define the app layout
app.layout = html.Div([
    html.H1("Vehicle Data Explorer"),
    
    # Categorical filters
    html.Div(categorical_filters, style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Numeric filters
    html.Div(numeric_filters, style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Graph area (using PCA coordinates)
    dcc.Graph(id='pca-scatter')
])

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
    
    # Create a 3D scatter plot using PCA coordinates (assuming they exist)
    if all(c in filtered_df.columns for c in ['P1', 'P2', 'P3']):
        fig = px.scatter_3d(
            filtered_df,
            x='P1', y='P2', z='P3',
            color='cluster',  # You can change the color mapping as desired
            hover_data=['Make', 'Model', 'Year', 'MSRP', 'value_score'],
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

if __name__ == '__main__':
    app.run(debug=True)
