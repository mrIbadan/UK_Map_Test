import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import folium_static

# Load GeoJSON file
with open("ITL3_JAN_2025_UK_BFE_1988110658864064311.geojson") as f:
    uk_geojson = json.load(f)

# Create dummy data for claims frequency and severity
uk_regions = [feature['properties']['ITL325NM'] for feature in uk_geojson['features']]
claims_frequency = np.random.randint(50, 200, len(uk_regions))
claims_severity = np.random.randint(1000, 5000, len(uk_regions))

uk_data = pd.DataFrame({
    'Region': uk_regions,
    'Claims Frequency': claims_frequency,
    'Claims Severity': claims_severity
})

def create_map(data_type):
    m = folium.Map(location=[55, -3], zoom_start=6)

    choropleth = folium.Choropleth(
        geo_data=uk_geojson,
        name="choropleth",
        data=uk_data,
        columns=["Region", data_type],
        key_on="feature.properties.ITL325NM",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=data_type
    ).add_to(m)

    tooltip = folium.GeoJsonTooltip(
        fields=['ITL325NM', data_type],
        aliases=['Region:', f'{data_type}:'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )

    choropleth.geojson.add_child(tooltip)

    return m

def main():
    st.title("UK Home Insurance Risk Analysis")

    data_type = st.selectbox(
        "Select data to display",
        ("Claims Frequency", "Claims Severity")
    )

    m = create_map(data_type)
    folium_static(m)

    st.write(f"This map shows the {data_type.lower()} for each region in the UK. Hover over a region to see its name and {data_type.lower()}.")

if __name__ == "__main__":
    main()
