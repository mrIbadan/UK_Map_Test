import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import requests
from io import BytesIO

# Set page config
st.set_page_config(page_title="UK Claims - Frequency and Severity Map", layout="wide")

# Load and display the logo in the top right corner
logo_url = "https://github.com/mrIbadan/UK_Map_Test/raw/main/Integra-Logo.jpg"
response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    st.image(logo, width=200)

# Title
st.title("UK Claims - Frequency and Severity Map")

# Load GeoJSON file from GitHub
@st.cache_data
def load_data():
    geojson_url = "https://raw.githubusercontent.com/mrIbadan/UK_Map_Test/main/Counties_and_Unitary_Authorities_December_2024_Boundaries_UK_BUC_-7342248301948489151.geojson"
    gdf = gpd.read_file(geojson_url)
    gdf = gdf.to_crs(epsg=4326)
    return gdf

gdf = load_data()

# Generate more diverse random data for demonstration
np.random.seed(42)  # for reproducibility

# Assign population with variation across regions
gdf['population'] = np.random.randint(50000, 2000000, size=len(gdf))

# Create region-based risk factors with variation
def assign_risk_factor(name):
    if "London" in name:
        return np.random.uniform(1.5, 3.0)  # Higher risk for London
    elif "Manchester" in name or "Birmingham" in name:
        return np.random.uniform(1.2, 2.5)  # Medium-high risk for large cities
    elif "Scotland" in name:
        return np.random.uniform(0.8, 1.5)  # Medium risk for Scotland
    elif "Wales" in name:
        return np.random.uniform(0.7, 1.3)  # Lower-medium risk for Wales
    else:
        return np.random.uniform(0.5, 2.0)  # Default variation for other regions

gdf['risk_factor'] = gdf['CTYUA24NM'].apply(assign_risk_factor)

# Generate claims frequency with significant variation
gdf['claim_frequency'] = np.random.poisson(lam=gdf['population'] * gdf['risk_factor'] / 5000)

# Generate claims severity with variation
gdf['claim_severity'] = np.random.gamma(shape=2, scale=gdf['risk_factor'] * 1500, size=len(gdf))

# Frequency Model (Poisson GLM)
X_freq = gdf[['population', 'risk_factor']]
y_freq = gdf['claim_frequency']
freq_model = PoissonRegressor(alpha=0.1)  # Adding some regularization
freq_model.fit(X_freq, y_freq)

# Severity Model (Gamma GLM)
X_sev = gdf[['population', 'risk_factor']]
y_sev = gdf['claim_severity']
sev_model = GammaRegressor(alpha=0.1)  # Adding some regularization
sev_model.fit(X_sev, y_sev)

# Predict and add to GeoDataFrame with variation included
gdf['predicted_frequency'] = freq_model.predict(X_freq)
gdf['predicted_severity'] = sev_model.predict(X_sev)

# Streamlit dropdown for selecting prediction type (Frequency or Severity)
prediction_type = st.selectbox(
    "Select Prediction Type",
    ("Claims Frequency", "Claims Severity")
)

# Create Folium map with proper variation included
m = folium.Map(location=[55, -3], zoom_start=6)

if prediction_type == "Claims Frequency":
    column_to_plot = 'predicted_frequency'
    legend_name = 'Predicted Claims Frequency'
    tooltip_alias = 'Claims Frequency'
else:
    column_to_plot = 'predicted_severity'
    legend_name = 'Predicted Claims Severity (£)'
    tooltip_alias = 'Claims Severity (£)'

# Create choropleth layer with color scale (Blue -> Green -> Yellow -> Red)
folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    name=legend_name,
    data=gdf,
    columns=['CTYUA24NM', column_to_plot],
    key_on='feature.properties.CTYUA24NM',
    fill_color='RdYlBu_r',  # Blue -> Yellow -> Red reversed for low-to-high risk coloring
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=legend_name,
).add_to(m)

# Add hover functionality to display region names and values dynamically
folium.GeoJson(
    gdf,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=['CTYUA24NM', column_to_plot],
        aliases=['County/UA Name', tooltip_alias],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
        localize=True,
    )
).add_to(m)

# Add layer control to toggle layers on/off
folium.LayerControl().add_to(m)

# Display the map in Streamlit with increased size for better visibility
st.markdown("## Interactive Map")
folium_static(m, width=1200, height=800)

# Display statistics for the selected prediction type (Frequency or Severity)
st.subheader(f"Statistics for {prediction_type}")
st.write(gdf[column_to_plot].describe())

# Display top 5 highest risk areas based on the selected prediction type
st.subheader(f"Top 5 Highest Risk Areas ({prediction_type})")
top_5 = gdf.nlargest(5, column_to_plot)[['CTYUA24NM', column_to_plot]]
st.write(top_5)

# Display bottom 5 lowest risk areas based on the selected prediction type
st.subheader(f"Top 5 Lowest Risk Areas ({prediction_type})")
bottom_5 = gdf.nsmallest(5, column_to_plot)[['CTYUA24NM', column_to_plot]]
st.write(bottom_5)
