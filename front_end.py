import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import requests
from requests.exceptions import RequestException
import json
import re
from typing import Dict, Any, Optional
import html as html_escape

# Security configuration
class SecurityConfig:
    MAX_STRING_LENGTH = 80
    MAX_NUMERIC_VALUE = 1000000
    ALLOWED_FUEL_TYPES = {'regular unleaded', 'premium unleaded', 'diesel', 'electric'}
    ALLOWED_TRANSMISSION_TYPES = {'AUTOMATIC', 'MANUAL'}
    ALLOWED_DRIVEN_WHEELS = {'front wheel drive', 'rear wheel drive', 'all wheel drive', 'four wheel drive'}
    ALLOWED_VEHICLE_SIZES = {'Compact', 'Midsize', 'Large'}
    ALLOWED_VEHICLE_STYLES = {'Sedan', 'SUV', 'Truck', 'Van', 'Wagon', 'Coupe', 'Convertible'}
    API_TIMEOUT = 10
    MAX_RETRIES = 3

def sanitize_input(value: str) -> str:
    """Sanitize string input to prevent XSS attacks."""
    if not isinstance(value, str):
        return str(value)
    # Remove potentially dangerous characters
    value = re.sub(r'[<>]', '', value)
    # Truncate to max length
    return value[:SecurityConfig.MAX_STRING_LENGTH]

def validate_numeric_input(value: float, field_name: str) -> tuple[bool, str]:
    """Validate numeric input values."""
    if not isinstance(value, (int, float)):
        return False, f"{field_name} must be a number"
    if value <= 0:
        return False, f"{field_name} must be positive"
    if value > SecurityConfig.MAX_NUMERIC_VALUE:
        return False, f"{field_name} exceeds maximum allowed value"
    return True, ""

def validate_car_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate car data before submission."""
    required_fields = [
        'Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
        'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
        'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity', 'MSRP'
    ]
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Sanitize string inputs
    for field in ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 
                 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']:
        data[field] = sanitize_input(data[field])
    
    # Validate categorical fields
    if data['Engine Fuel Type'] not in SecurityConfig.ALLOWED_FUEL_TYPES:
        return False, "Invalid engine fuel type"
    if data['Transmission Type'] not in SecurityConfig.ALLOWED_TRANSMISSION_TYPES:
        return False, "Invalid transmission type"
    if data['Driven_Wheels'] not in SecurityConfig.ALLOWED_DRIVEN_WHEELS:
        return False, "Invalid driven wheels type"
    if data['Vehicle Size'] not in SecurityConfig.ALLOWED_VEHICLE_SIZES:
        return False, "Invalid vehicle size"
    if data['Vehicle Style'] not in SecurityConfig.ALLOWED_VEHICLE_STYLES:
        return False, "Invalid vehicle style"
    
    # Validate numeric fields
    numeric_fields = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 
                     'highway MPG', 'city mpg', 'Popularity', 'MSRP']
    for field in numeric_fields:
        is_valid, error_msg = validate_numeric_input(float(data[field]), field)
        if not is_valid:
            return False, error_msg
    
    return True, ""

def make_api_request(method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Make an API request with retry logic and error handling."""
    for attempt in range(SecurityConfig.MAX_RETRIES):
        try:
            response = requests.request(
                method,
                url,
                timeout=SecurityConfig.API_TIMEOUT,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt == SecurityConfig.MAX_RETRIES - 1:
                raise
            continue
        except requests.exceptions.RequestException as e:
            raise
    return None

# Load initial data with better error handling
def load_initial_data():
    try:
        data = make_api_request('GET', "http://localhost:5000/cars")
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading initial data: {str(e)}")
        return pd.DataFrame()

# Load initial data
df = load_initial_data()

# Define which columns are categorical and which are numeric
categorical_columns = [
    'Make', 'Model', 'Engine Fuel Type', 'Engine Cylinders',
    'Transmission Type', 'Driven_Wheels', 'Number of Doors',
    'Market Category', 'Vehicle Size', 'Vehicle Style'
]
numeric_columns = ['Engine HP', 'highway MPG', 'city mpg', 'MSRP', 'value_score', 'meta_value_score',
                'cluster', 'meta_cluster']

# Build the "Add Car" modal form.
# Define the user-input fields:
new_car_fields = [
    ('Make', 'text'),
    ('Model', 'text'),
    ('Year', 'number'),
    ('Engine Fuel Type', 'text'),
    ('Engine HP', 'number'),
    ('Engine Cylinders', 'number'),
    ('Transmission Type', 'text'),
    ('Driven_Wheels', 'text'),
    ('Number of Doors', 'number'),
    ('Market Category', 'text'),
    ('Vehicle Size', 'text'),
    ('Vehicle Style', 'text'),
    ('highway MPG', 'number'),
    ('city mpg', 'number'),
    ('Popularity', 'number'),
    ('MSRP', 'number')
]

# Create form groups for each field
# For categorical fields, we use a dropdown populated with unique values from df.
add_car_form = []
for field, typ in new_car_fields:
    if field in categorical_columns:
        options = [{'label': str(val), 'value': val} for val in sorted(df[field].dropna().unique())]
        add_car_form.append(
            dbc.CardGroup([
                dbc.Label(field),
                dcc.Dropdown(
                    id=f"input-{field.replace(' ', '_').lower()}",
                    options=options,
                    placeholder=f"Select {field}",
                    style={'width': '100%'}
                )
            ], className="mb-3")
        )
    else:
        # Numeric field: use a numeric input
        add_car_form.append(
            dbc.CardGroup([
                dbc.Label(field),
                dbc.Input(
                    id=f"input-{field.replace(' ', '_').lower()}",
                    type=typ,
                    placeholder=f"Enter {field}"
                )
            ], className="mb-3")
        )


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
        ], style={'marginBottom': '15px'}, className='mb-3')
    )

# Create filter components for numeric columns
numeric_filters = []
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    min_val = pd.to_numeric(df[col].min(), errors='coerce')
    max_val = pd.to_numeric(df[col].max(), errors='coerce')
    # A basic step calculation; adjust this as needed
    step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 1
    numeric_filters.append(
        html.Div([
            html.Label(f"Select {col} Range:"),
            dcc.RangeSlider(
                id=f"{col.replace(' ', '_').lower()}-slider",
                min=min_val,
                max=max_val,
                step=(max_val - min_val) / 1000 if (max_val - min_val) > 0 else 1,
                value=[min_val, max_val],
                marks={
                    int(min_val): str(int(min_val)),
                    int(max_val): str(int(max_val))
                }
            )
        ], style={'marginBottom': '25px'}, className='mb-4')
    )

all_filters = categorical_filters + numeric_filters

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# Define the app layout
app.layout = dbc.Container([

    dcc.Store(id='update-trigger', data=0),

    dbc.NavbarSimple(
        brand="Vehicle Data Explorer",
        color="primary",
        dark=True,
        sticky="top"
    ),
    # Filter button with dynamic z-index
    dbc.Button(
        "Filters",
        id="open-offcanvas",
        n_clicks=0,
        color="secondary",
        style={"position": "fixed", "top": "80px", "left": "20px", "zIndex": 1100},
        className="filter-button"
    ),
    dbc.Offcanvas(
        html.Div(all_filters, style={'padding': '10px'}),
        id="offcanvas",
        title="Filter Options",
        is_open=False,
        placement="start",
        backdrop=True,
        style={"zIndex": 1200}  # Higher z-index than the button
    ),
    # "Add Car" button with dynamic z-index
    dbc.Button(
        "Add Car",
        id="open-add-car",
        n_clicks=0,
        color="success",
        style={"position": "fixed", "top": "80px", "right": "20px", "zIndex": 1100},
        className="add-car-button"
    ),
    # Add Car modal form with higher z-index
    dbc.Modal(
        [
            dbc.ModalHeader("Add a New Car: *All fields must be filled*"),
            dbc.ModalBody(dbc.Form(add_car_form)),
            dbc.ModalFooter([
                dbc.Button("Submit", id="submit-car", color="primary", className="mr-2"),
                dbc.Button("Close", id="close-add-car", color="secondary")
            ])
        ],
        id="add-car-modal",
        is_open=False,
        style={"zIndex": 1200}  # Higher z-index than the button
    ),
    # Div to display status messages (e.g., submission success)
    html.Div(id="add-car-status", className="mt-3"),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='pca-scatter',
                style={'height': '50vh', 'width': '100%'}  # Graph occupies the top half of the viewport
            ),
            width=12
        )
    ], className="mt-5"),
    dbc.Row([
        dbc.Col(
            html.Div(
                id='car-details',
                style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }
            ),
            width=12
        )
    ], className="mt-3")
], fluid=True)

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

# Toggle Add Car modal open/close
@app.callback(
    Output("add-car-modal", "is_open"),
    [Input("open-add-car", "n_clicks"),
     Input("close-add-car", "n_clicks"),
     Input("submit-car", "n_clicks")],
    State("add-car-modal", "is_open")
)
def toggle_add_car_modal(open_clicks, close_clicks, submit_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id in ["open-add-car", "close-add-car", "submit-car"]:
            return not is_open
    return is_open

# Callback to handle new car submission with better error handling
@app.callback(
    [Output("add-car-status", "children"),
     Output("update-trigger", "data")],
    Input("submit-car", "n_clicks"),
    [State(f"input-{field.replace(' ', '_').lower()}", "value") for field, _ in new_car_fields],
    State("update-trigger", "data"),
    prevent_initial_call=True
)
def handle_new_car(submit_clicks, *args):
    update_val = args[-1]
    input_values = args[:-1]
    
    if submit_clicks:
        # Validate all required fields are filled
        if any(value is None for value in input_values):
            return dbc.Alert("All fields must be filled", color="danger"), update_val
            
        # Create a dict for the new car from the form inputs
        new_car = {field: value for (field, _), value in zip(new_car_fields, input_values)}
        
        # Validate and sanitize input data
        is_valid, error_message = validate_car_data(new_car)
        if not is_valid:
            return dbc.Alert(error_message, color="danger"), update_val

        try:
            # Post new vehicle data to Flask server with retry logic
            response_data = make_api_request('POST', "http://localhost:5000/cars", json=new_car)
            if response_data is None:
                return dbc.Alert("Failed to submit new car after multiple attempts", color="danger"), update_val
                
            if 'error' in response_data:
                return dbc.Alert(response_data['error'], color="danger"), update_val
                
            return dbc.Alert("New car submitted successfully", color="success"), update_val + 1
            
        except Exception as e:
            return dbc.Alert(f"Error submitting new car: {str(e)}", color="danger"), update_val
            
    return "", update_val

# Create a callback that takes the values from each filter
@app.callback(
    Output('pca-scatter', 'figure'),
    [Input('update-trigger', 'data')] +
    [Input(f"{col.replace(' ', '_').lower()}-dropdown", 'value') for col in categorical_columns] +
    [Input(f"{col.replace(' ', '_').lower()}-slider", 'value') for col in numeric_columns]
)
def update_graph(update_trigger, *args):
    # Get updated dataset from Flask server with retry logic
    try:
        data = make_api_request('GET', "http://localhost:5000/cars")
        if data is None:
            updated_df = df.copy()
        else:
            updated_df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error updating data: {str(e)}")
        updated_df = df.copy()

    for col in numeric_columns:
        updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce').fillna(0)

    # Separate the inputs: first n are categorical, then numeric inputs follow
    num_cat = len(categorical_columns)
    cat_values = args[:num_cat]
    num_values = args[num_cat:]
    
    # Apply filters for categorical columns
    for i, col in enumerate(categorical_columns):
        selected = cat_values[i]
        if selected and len(selected) > 0:
            updated_df = updated_df[updated_df[col].isin(selected)]
    
    # Apply filters for numeric columns
    for i, col in enumerate(numeric_columns):
        slider_range = num_values[i]
        if slider_range:
            updated_df = updated_df[(updated_df[col] >= slider_range[0]) & 
                                      (updated_df[col] <= slider_range[1])]
            
    updated_df = updated_df.reset_index()
    
    # Create a 3D scatter plot using PCA coordinates
    if all(c in updated_df.columns for c in ['P1', 'P2', 'P3']):
        fig = px.scatter_3d(
            updated_df,
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
            updated_df,
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
    
    try:
        # Extract the custom data (the index) from the clicked point
        point_index = clickData['points'][0]['customdata'][0]
        # Retrieve the corresponding row from the original DataFrame
        car_data = df.loc[point_index]
        
        # Build a table to display all of the car's data with XSS prevention
        table_rows = []
        for col, val in car_data.items():
            # Handle NaN values and escape HTML
            if pd.isna(val):
                val = "N/A"
            else:
                val = html_escape.escape(str(val))
            table_rows.append(
                html.Tr([
                    html.Td(html.B(html_escape.escape(col)), 
                           style={'border': '1px solid #ccc', 'padding': '5px'}),
                    html.Td(val, style={'border': '1px solid #ccc', 'padding': '5px'})
                ])
            )
        
        table = dbc.Table(table_rows, bordered=True, striped=True, hover=True, responsive=True)
        return table
    except KeyError:
        return dbc.Alert("Error: Selected data point not found", color="danger")
    except Exception as e:
        return dbc.Alert(f"Error displaying car details: {str(e)}", color="danger")

# Add callback to handle button visibility
@app.callback(
    [Output("open-offcanvas", "style"),
     Output("open-add-car", "style")],
    [Input("offcanvas", "is_open"),
     Input("add-car-modal", "is_open")]
)
def update_button_styles(offcanvas_open, modal_open):
    filter_button_style = {
        "position": "fixed",
        "top": "80px",
        "left": "20px",
        "zIndex": 900 if offcanvas_open else 1100  # Lower z-index when menu is open
    }
    add_car_button_style = {
        "position": "fixed",
        "top": "80px",
        "right": "20px",
        "zIndex": 900 if modal_open else 1100  # Lower z-index when menu is open
    }
    return filter_button_style, add_car_button_style

if __name__ == '__main__':
    app.run(debug=True)
