import streamlit as st
import streamlit.components.v1 as components

def main():
    st.title("UK Risk Map")

    # Load and display the HTML map
    html_file_path = "maps.html"  # Ensure this file is in the same directory as app.py

    with open(html_file_path, 'r') as f:
        html_content = f.read()

    # Render HTML with sufficient height
    components.html(html_content, height=600, scrolling=True)

if __name__ == "__main__":
    main()
