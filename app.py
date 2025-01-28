import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import folium_static
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
        return np.random.uniform(1.5, 3.0)
    elif "Manchester" in name or "Birmingham" in name:
        return np.random.uniform(1.2, 2.5)
    elif "Scotland" in name:
        return np.random.uniform(0.8, 1.5)
    elif "Wales" in name:
        return np.random.uniform(0.7, 1.3)
    else:
        return np.random.uniform(0.5, 2.0)

gdf['risk_factor'] = gdf['CTYUA24NM'].apply(assign_risk_factor)

# Generate claims frequency and severity with significant variation
gdf['claim_frequency'] = np.random.negative_binomial(n=10, p=0.5, size=len(gdf)) * gdf['risk_factor']
gdf['claim_severity'] = np.random.gamma(shape=2, scale=gdf['risk_factor'] * 1000, size=len(gdf))

# Create features for the model
gdf['area'] = gdf.geometry.area
gdf['perimeter'] = gdf.geometry.length
X = gdf[['population', 'risk_factor', 'area', 'perimeter']]

# Frequency Model (Random Forest)
freq_model = RandomForestRegressor(n_estimators=100, random_state=42)
freq_model.fit(X, gdf['claim_frequency'])

# Severity Model (Random Forest)
sev_model = RandomForestRegressor(n_estimators=100, random_state=42)
sev_model.fit(X, gdf['claim_severity'])

# Predict and add to GeoDataFrame
gdf['predicted_frequency'] = freq_model.predict(X)
gdf['predicted_severity'] = sev_model.predict(X)

# Streamlit dropdown for selecting prediction type
prediction_type = st.selectbox(
    "Select Prediction Type",
    ("Claims Frequency", "Claims Severity")
)

# Create Folium map
m = folium.Map(location=[55, -3], zoom_start=6)

if prediction_type == "Claims Frequency":
    column_to_plot = 'predicted_frequency'
    legend_name = 'Predicted Claims Frequency'
    tooltip_alias = 'Claims Frequency'
else:
    column_to_plot = 'predicted_severity'
    legend_name = 'Predicted Claims Severity (£)'
    tooltip_alias = 'Claims Severity (£)'

# Create choropleth layer
folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    name=legend_name,
    data=gdf,
    columns=['CTYUA24NM', column_to_plot],
    key_on='feature.properties.CTYUA24NM',
    fill_color='RdYlBu_r',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=legend_name,
).add_to(m)

# Add hover functionality
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

# Add layer control
folium.LayerControl().add_to(m)

# Display the map in Streamlit
st.markdown("## Interactive Map")
folium_static(m, width=1200, height=800)

# Display statistics
st.subheader(f"Statistics for {prediction_type}")
st.write(gdf[column_to_plot].describe())

# Display top 5 highest risk areas
st.subheader(f"Top 5 Highest Risk Areas ({prediction_type})")
top_5 = gdf.nlargest(5, column_to_plot)[['CTYUA24NM', column_to_plot]]
st.write(top_5)

# Display bottom 5 lowest risk areas
st.subheader(f"Top 5 Lowest Risk Areas ({prediction_type})")
bottom_5 = gdf.nsmallest(5, column_to_plot)[['CTYUA24NM', column_to_plot]]
st.write(bottom_5)
