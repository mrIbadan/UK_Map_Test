import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="UK Claims Prediction Map", layout="wide")

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
gdf['population'] = np.random.randint(10000, 1000000, size=len(gdf))

# Create region-based risk factors
def assign_risk_factor(name):
    if "London" in name:
        return np.random.uniform(1.5, 2.0)
    elif "Manchester" in name or "Birmingham" in name:
        return np.random.uniform(1.2, 1.8)
    elif "Scotland" in name:
        return np.random.uniform(0.8, 1.2)
    elif "Wales" in name:
        return np.random.uniform(0.7, 1.1)
    else:
        return np.random.uniform(0.5, 1.5)

gdf['risk_factor'] = gdf['CTYUA24NM'].apply(assign_risk_factor)
gdf['claim_frequency'] = np.random.poisson(lam=gdf['population'] * gdf['risk_factor'] / 5000)
gdf['claim_severity'] = np.random.gamma(shape=2, scale=gdf['risk_factor'] * 1000, size=len(gdf))

# Frequency Model (Poisson GLM)
X = gdf[['population', 'risk_factor']]
y_freq = gdf['claim_frequency']
freq_model = PoissonRegressor()
freq_model.fit(X, y_freq)

# Severity Model (Gamma GLM)
y_sev = gdf['claim_severity']
sev_model = GammaRegressor()
sev_model.fit(X, y_sev)

# Predict and add to GeoDataFrame
gdf['predicted_frequency'] = freq_model.predict(X)
gdf['predicted_severity'] = sev_model.predict(X)

# Streamlit app
st.title("UK Claims Prediction Map")

# Dropdown for selecting prediction type
prediction_type = st.selectbox(
    "Select Prediction Type",
    ("Claims Frequency", "Claims Severity")
)

# Create Folium map
m = folium.Map(location=[55, -3], zoom_start=6)

# Determine which column to use based on selection
if prediction_type == "Claims Frequency":
    column_to_plot = 'predicted_frequency'
    legend_name = 'Predicted Claims Frequency'
    tooltip_alias = 'Claims Frequency'
else:
    column_to_plot = 'predicted_severity'
    legend_name = 'Predicted Claims Severity'
    tooltip_alias = 'Claims Severity (£)'

# Create choropleth layer
folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    name=legend_name,
    data=gdf,
    columns=['CTYUA24NM', column_to_plot],
    key_on='feature.properties.CTYUA24NM',
    fill_color='RdYlBu_r',  # This gives a blue-yellow-red scale
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=legend_name,
    bins=4,  # This will create 4 bins for our 4 risk levels
).add_to(m)

# Add hover functionality
folium.GeoJson(
    gdf,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=['CTYUA24NM', column_to_plot],
        aliases=['County/UA Name', tooltip_alias],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
        localize=True,  # This will format numbers according to locale
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display the map in Streamlit
st.markdown("## Interactive Map")
folium_static(m, width=1200, height=800)  # Increased size

# Display some statistics
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
