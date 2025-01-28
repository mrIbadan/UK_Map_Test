import streamlit as st
import geopandas as gpd
import numpy as np
from sklearn.linear_model import PoissonRegressor
import folium
from streamlit_folium import folium_static

# Load GeoJSON file from URL
url = "https://raw.githubusercontent.com/mrIbadan/UK_Map_Test/main/Counties_and_Unitary_Authorities_December_2024_Boundaries_UK_BUC_-7342248301948489151.geojson"
gdf = gpd.read_file(url)

# Ensure the GeoDataFrame is in the correct CRS for Folium (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)

# Generate random data for demonstration
gdf['population'] = np.random.randint(10000, 1000000, size=len(gdf))
gdf['claim_frequency'] = np.random.poisson(lam=gdf['population'] / 10000)
gdf['claim_severity'] = np.random.uniform(low=100, high=10000, size=len(gdf))  # Example severity data

# Frequency Model (Poisson GLM)
X_freq = gdf[['population']]
y_freq = gdf['claim_frequency']
freq_model = PoissonRegressor()
freq_model.fit(X_freq, y_freq)
gdf['predicted_frequency'] = freq_model.predict(X_freq)

# Severity Model (Poisson GLM)
X_severity = gdf[['population']]
y_severity = gdf['claim_severity']
severity_model = PoissonRegressor()
severity_model.fit(X_severity, y_severity)
gdf['predicted_severity'] = severity_model.predict(X_severity)

# Streamlit app layout
st.title("Claims Frequency and Severity Map")
option = st.selectbox("Select Metric to Display", ["Claims Frequency", "Claims Severity"])

# Create Folium map
m = folium.Map(location=[55, -3], zoom_start=6)

if option == "Claims Frequency":
    # Create choropleth layer for Claims Frequency
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        name='Predicted Claim Frequency',
        data=gdf,
        columns=['CTYUA24NM', 'predicted_frequency'],
        key_on='feature.properties.CTYUA24NM',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Predicted Claim Frequency'
    ).add_to(m)
    
    tooltip_fields = ['CTYUA24NM', 'predicted_frequency']
    
else:
    # Create choropleth layer for Claims Severity
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        name='Predicted Claim Severity',
        data=gdf,
        columns=['CTYUA24NM', 'predicted_severity'],
        key_on='feature.properties.CTYUA24NM',
        fill_color='BuGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Predicted Claim Severity'
    ).add_to(m)
    
    tooltip_fields = ['CTYUA24NM', 'predicted_severity']

# Add hover functionality
folium.GeoJson(
    gdf,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=['County/UA Name', 'Predicted Value'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Render the map in Streamlit
folium_static(m)
