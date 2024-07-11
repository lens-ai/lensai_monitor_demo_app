import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import re
import os
from sketchmanager import SketchManager
from quantilemetrics import QuantileMetrics
from datasketches import kll_floats_sketch
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from datetime import datetime
from plotly.subplots import make_subplots
from PIL import Image
import argparse
import math


st.set_page_config(page_title="LensAI Monitor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>LensAI Observer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6C63FF;'>Model and Data Observality</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Monitoring</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
""", unsafe_allow_html=True)


dist_placeholder = st.empty()

def get_histogram(sketch, num_splits=30):
    """
    Reads a binary file, deserializes the content, and extracts the PMF.

    Args:
        filename: Path to the binary file.
        num_splits: Number of splits for the PMF (default: 30).

    Returns:
        A tuple containing x-axis values and the PMF.
    """
    if sketch.is_empty():
        return None, None
    xmin = sketch.get_min_value()
    try:
        step = (sketch.get_max_value() - xmin) / num_splits
    except ZeroDivisionError:
        print(f"Error: num_splits should be non-zero for file {filename}")
        return None, None
    if step == 0:
        step = 0.01

    splits = [xmin + (i * step) for i in range(0, num_splits)]
    pmf = sketch.get_pmf(splits)
    x = splits + [sketch.get_max_value()]  # Append max value for x-axis

    return x, pmf

def plot_histograms(time_series_stats, selected_timestamp, reference_stats=None):
    figures = []
    for metric, sub_metrics in next(iter(time_series_stats.values())).items():
        fig = go.Figure()

        if isinstance(sub_metrics, dict):
            for sub_metric in sub_metrics.keys():
                for time_point, stats in time_series_stats.items():
                    if time_point == selected_timestamp:
                        x, pmf = get_histogram(stats[metric][sub_metric])
                        fig.add_trace(go.Bar(
                            x=x, y=pmf, name="{} - {} ({})".format(metric, sub_metric, datetime.utcfromtimestamp(int(time_point)).strftime('%Y-%m-%dT%H:%M:%SZ')),
                            marker_color='blue'))

                if reference_stats:
                    last_reference = list(reference_stats.keys())[0]
                    if metric in reference_stats[last_reference] and sub_metric in reference_stats[last_reference][metric]:
                        ref_sketch = reference_stats[last_reference][metric][sub_metric]
                        ref_x, ref_pmf = get_histogram(ref_sketch)
                        fig.add_trace(go.Bar(
                            x=ref_x, y=ref_pmf, name=f"{metric} - {sub_metric} (Reference)",
                            marker_color='orange', opacity=0.6))
        else:
            for time_point, stats in time_series_stats.items():
                if time_point == selected_timestamp:
                    x, pmf = get_histogram(stats[metric])
                    fig.add_trace(go.Bar(
                        x=x, y=pmf, name="{} ({})".format(metric, datetime.utcfromtimestamp(int(time_point)).strftime('%Y-%m-%dT%H:%M:%SZ')),
                        marker_color='blue'))

            if reference_stats:
                last_reference = list(reference_stats.keys())[0]
                if metric in reference_stats[last_reference]:
                    ref_sketch = reference_stats[last_reference][metric]
                    ref_x, ref_pmf = get_histogram(ref_sketch)
                    fig.add_trace(go.Bar(
                        x=ref_x, y=ref_pmf, name=f"{metric} (Reference)",
                        marker_color='orange', opacity=0.6))

        fig.update_layout(
            title=f"Metric: {metric}",
            xaxis_title='Value',
            yaxis_title='Probability',
            barmode='overlay',
            showlegend=True
        )

        figures.append(fig)

    return figures

def convert_timestamp(timestamp):
    try:
        # Handle both 10-digit and 16-digit timestamps
        if len(timestamp) == 10:
            return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
        elif len(timestamp) == 16:
            return datetime.fromtimestamp(int(timestamp) / 1e6).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        st.error(f"Error converting timestamp {timestamp}: {e}")
        return timestamp

# Function to display images with pagination
def plot_images(sensor_sample_images, selected_sensor, selected_timestamp):
    st.text("Sampled images based on the metric thresholds set for each sensor, where the model is most uncertain")
    try:
        images_info = sensor_sample_images[selected_sensor][selected_timestamp]
    except:
        st.write("No Sampled Images")
        return
    num_columns = 2  # Number of columns in each row
    images_per_page = 10  # Number of images per page

    for metric, paths in images_info.items():
        with st.expander(metric):
            if isinstance(paths, dict):
                all_paths = [(sub_metric, path) for sub_metric, path_list in paths.items() for path in path_list]
            else:
                all_paths = [(None, path) for path in paths]

            num_images = len(all_paths)
            num_pages = (num_images + images_per_page - 1) // images_per_page  # Calculate number of pages

            # Create a selection box for the page number
            page_number = st.selectbox(f"Select Page for {metric}", range(1, num_pages + 1))

            # Calculate start and end indices for the images to be displayed on the current page
            start_idx = (page_number - 1) * images_per_page
            end_idx = min(start_idx + images_per_page, num_images)

            fig = make_subplots(
                rows=(images_per_page + num_columns - 1) // num_columns,
                cols=num_columns,
                subplot_titles=[
                    f"{sub_metric if sub_metric else ''} - {convert_timestamp(os.path.basename(path).split('_')[1].split('.')[0])}"
                    for sub_metric, path in all_paths[start_idx:end_idx]
                ]
            )

            for idx in range(start_idx, end_idx):
                sub_metric, path = all_paths[idx]
                row = (idx - start_idx) // num_columns + 1
                col = (idx - start_idx) % num_columns + 1
                image = Image.open(path)
                fig.add_trace(
                    go.Image(z=image),
                    row=row, col=col
                )

            fig.update_layout(height=800, width=1000, title_text=f"{metric} Images (Page {page_number})")
            st.plotly_chart(fig)

def plot_distance_metrics_time_series(distance_metrics_over_time):
    fig = go.Figure()
    for time_point, metrics in distance_metrics_over_time.items():
        for metric, sub_metrics in metrics.items():
            if isinstance(sub_metrics, dict):
                for sub_metric, distance in sub_metrics.items():
                    fig.add_trace(go.Scatter(
                        x=[datetime.utcfromtimestamp(int(time_point)).strftime('%Y-%m-%dT%H:%M:%SZ')], y=[distance], mode='lines+markers', name=f"{metric} - {sub_metric}",
                        line=dict(shape='linear')))
            else:
                fig.add_trace(go.Scatter(
                    x=[datetime.utcfromtimestamp(int(time_point)).strftime('%Y-%m-%dT%H:%M:%SZ')], y=[sub_metrics], mode='lines+markers', name=f"{metric}",
                    line=dict(shape='linear')))

    fig.update_layout(
        title="Distance Metrics Over Time",
        xaxis_title="Time",
        yaxis_title="Distance",
        height=600,
        width=1000
    )
    return fig

def compute_distance_metrics(stats1, stats2, metric_name):
    def recursive_distance_computation(dict1, dict2, path=[]):
        distance_metrics = {}
        for key, value in dict1.items():
            if key in dict2:
                if isinstance(value, dict) and isinstance(dict2[key], dict):
                    distance_metrics[key] = recursive_distance_computation(value, dict2[key], path + [key])
                elif isinstance(value, kll_floats_sketch) and isinstance(dict2[key], kll_floats_sketch):
                    # Compute distance metric between sketches
                    x1, pmf1 = get_histogram(value)
                    x2, pmf2 = get_histogram(dict2[key])
                    metrics = QuantileMetrics(pmf1, x1, pmf2, x2)
                    dist = getattr(metrics, metric_name)()
                    distance_metrics[key] = dist
                else:
                    print(f"Skipping key {key} at path {'.'.join(path)} due to incompatible types or missing sub-dictionary.")
        return distance_metrics
    st.subheader("Distance Metrics between reference and aggregated sensor distributions")
    return recursive_distance_computation(stats1, stats2)

# Function to compute and display distance metrics between sketches
def display_distance_metrics(stats1, stats2, selected_metric):
    distance_metrics = {}
    last_reference = list(stats1.keys())[0]
    for timestamp, metrics in stats2.items():
        if timestamp not in stats2:
            distance_metrics[timestamp] = {}
        distance_metrics[timestamp] = compute_distance_metrics(stats1[last_reference], metrics, selected_metric)
    dist_placeholder.subheader("Distance Metrics")
    distance_plot = plot_distance_metrics_time_series(distance_metrics)
    dist_placeholder.plotly_chart(distance_plot)

def extract_info(file_path):
    pattern_sensors = re.compile(r'/data/(sensor\d+)/(\d+)/(modelstats|imagestats|samples)/(.+?)(?:_(\d+))?\.bin')
    pattern_reference = re.compile(r'/data/reference/(\d+)/(modelstats|imagestats|samples)/(.+?)(?:_(\d+))?\.bin')
    pattern_sampledimages = re.compile(r'/data/(sensor\d+)/(\d+)/(modelstats|imagestats|samples)/([A-Z]+)(?:_(\d+))?_(\d+)\.png')
    match_sensors = pattern_sensors.search(file_path)
    match_reference = pattern_reference.search(file_path)
    match_samples = pattern_sampledimages.search(file_path)
    if match_sensors:
        sensor_id, timestamp, stat_type, metric, metric_subtype = match_sensors.groups()
        metric_subtype = str(metric_subtype) if metric_subtype else ''
        return sensor_id, timestamp, stat_type, metric, metric_subtype
    elif match_reference:
        stat_type, timestamp, metric, metric_subtype = match_reference.groups()
        metric_subtype = str(metric_subtype) if metric_subtype else ''
        return 'reference', stat_type, timestamp, metric, metric_subtype
    elif match_samples:
        sensor_id, timestamp, stat_type, metric, metric_subtype, generate_time = match_samples.groups()
        return sensor_id, timestamp, stat_type, metric, metric_subtype
    return None, None, None, None, None

# Streamlit app
def main(data_dir):
    if data_dir is None or "":
        data_dir = "./data"
    elif not os.path.exists(data_dir):
        st.write("Path doesn't exists")
        raise Exception('general exceptions not caught by specific handling')
    sketch_manager = SketchManager()
    last_updated = str(int(datetime.timestamp(datetime.now())))
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bin') or file.endswith('.png'):
                file_path = os.path.join(root, file)
                sensor_id, file_timestamp, current_file_type, current_metric, current_sub_metric = extract_info(file_path)
                if sensor_id == 'reference':
                    sketch_manager.add_reference_sketch(last_updated, current_file_type, current_metric, current_sub_metric, file_path)
                elif sensor_id:
                    sketch_manager.add_sketch(sensor_id, last_updated, file_timestamp, current_file_type, current_metric, current_sub_metric, file_path)
    selected_stat = st.sidebar.radio("Select Profile", ['Image Profile', 'Model Profile', 'Sensor Profile'])
    if selected_stat == 'Model Profile':
        handle_model_stats(sketch_manager)
    elif selected_stat == 'Image Profile':
        handle_image_stats(sketch_manager)
    elif selected_stat == 'Sensor Profile':
        handle_sensor_stats(sketch_manager)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <a href='https://your-website-url.com' style='color: #6C63FF; text-decoration: none;'>Visit our website</a>
        <br>
        <p>© 2024 LensAI. <h3 style='text-align: center; color: #6C63FF;'>Made with 💜 from Berlin</h3>. All rights reserved.</p>
        <p>Licensed under the MIT License.</p>
    </div>
    """, unsafe_allow_html=True)

def handle_image_stats(sketch_manager):
    st.markdown("<h2 style='color: #6C63FF;'>Data Image Statistics</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #4B4B4B;'>Analyze aggregated image data statistics of all the sensors.
    Choose from various distance metrics to find the drift between the baseline and realtime data. </p>
    <hr style='border-top: 2px solid #6C63FF;'>
    """, unsafe_allow_html=True)

    available_metrics = QuantileMetrics.available_metrics()
    default_metric = "psi"
    selected_metric = st.sidebar.selectbox("Select Distance Metric", available_metrics, index=available_metrics.index(default_metric))
    
    st.sidebar.text("")

    time_series_stats = sketch_manager.image_stats
    reference_stats = sketch_manager.image_stats_ref
    display_distance_metrics(reference_stats, time_series_stats, selected_metric)

    timestamps = list(time_series_stats.keys())
    if len(timestamps) >= 1:
        selected_timestamp = timestamps[0] #st.sidebar.select_slider("Select Timestamp", options=timestamps, value=timestamps[0])
        selected_timestamp_str = datetime.utcfromtimestamp(int(selected_timestamp)).strftime('%Y-%m-%dT%H:%M:%SZ')  # Convert selected timestamp for display
        st.sidebar.text(f"Aggregation Last Updated: {selected_timestamp_str}")
    else:
        st.sidebar.text("No timestamps available")
        return  # Exit the function early if no timestamps are available

    st.text("Histograms of the image data metrics")
    # Plot histograms if they haven't been plotted yet or selected_metric has changed
    figures = plot_histograms(time_series_stats, selected_timestamp, reference_stats)
    # Display figures in a grid layout
    num_columns = 2  # Number of columns in each row
    rows = [figures[i:i + num_columns] for i in range(0, len(figures), num_columns)]

    for row in rows:
        cols = st.columns(num_columns)
        for col, fig in zip(cols, row):
            col.plotly_chart(fig)


def handle_sensor_stats(sketch_manager):
    st.markdown("<h2 style='color: #6C63FF;'>Sensor Statistics</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #4B4B4B;'>Dive into the sensor level statistics with detailed visualizations of sensor level metrics and sampled data.
    Select a sensor and a timestamp to view the associated images.</p>
    <hr style='border-top: 2px solid #6C63FF;'>
    """, unsafe_allow_html=True)
 
    # Dropdown to select a sensor
    sensor_ids = list(sketch_manager.sensor_image_stats.keys())
    selected_sensor = None
    selected_sensor = st.sidebar.selectbox("Select Sensor", sensor_ids, index=0)

    if selected_sensor != "None":
        st.sidebar.markdown(f"**Selected Sensor: {selected_sensor}**")

        last_updated_datetime = datetime.utcfromtimestamp(int(list(sketch_manager.sensor_image_stats[selected_sensor].keys())[-1])).strftime('%Y-%m-%d %H:%M:%S')
        st.sidebar.markdown(f"<span style='color: green;'>{last_updated_datetime}</span>", unsafe_allow_html=True)

        timestamps = list(sketch_manager.sensor_image_stats[selected_sensor].keys())
        timestamps_str = [datetime.utcfromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]
        t_index = 0
        if len(timestamps) > 1:
            selected_timestamp_str = st.sidebar.select_slider("Select Timestamp", options=timestamps_str, value=timestamps_str[t_index])
            t_index = timestamps_str.index(selected_timestamp_str)
            selected_timestamp = timestamps[t_index]
        else:
            selected_timestamp = timestamps[t_index]
            st.sidebar.text(f"Only one timestamp available: {selected_timestamp_str}")
          
        st.subheader(f"Sensor: {selected_sensor}, Timestamp: {selected_timestamp_str}")
        # Plot Image Stats
        filtered_time_series_stats = {k: v for k, v in sketch_manager.sensor_image_stats.items() if k == selected_sensor}
        figures = plot_histograms(filtered_time_series_stats[selected_sensor], selected_timestamp)
        # Display figures in a grid layout
        num_columns = 2  # Number of columns in each row
        rows = [figures[i:i + num_columns] for i in range(0, len(figures), num_columns)]

        for row in rows:
            cols = st.columns(num_columns)
            for col, fig in zip(cols, row):
                col.plotly_chart(fig)

        # Plot Model Stats
        filtered_time_series_stats = {k: v for k, v in sketch_manager.sensor_model_stats.items() if k == selected_sensor}
        figures = plot_histograms(filtered_time_series_stats[selected_sensor], selected_timestamp)
        # Display figures in a grid layout
        num_columns = 2  # Number of columns in each row
        rows = [figures[i:i + num_columns] for i in range(0, len(figures), num_columns)]

        for row in rows:
            cols = st.columns(num_columns)
            for col, fig in zip(cols, row):
                col.plotly_chart(fig)

        # Plot Sample Stats
        filtered_time_series_stats = {k: v for k, v in sketch_manager.sensor_sample_stats.items() if k == selected_sensor}
        figures = plot_histograms(filtered_time_series_stats[selected_sensor], selected_timestamp)
        # Display figures in a grid layout
        num_columns = 2  # Number of columns in each row
        rows = [figures[i:i + num_columns] for i in range(0, len(figures), num_columns)]
        for row in rows:
            cols = st.columns(num_columns)
            for col, fig in zip(cols, row):
                col.plotly_chart(fig)
        if selected_sensor in sketch_manager.sensor_sample_images:
            plot_images(sketch_manager.sensor_sample_images, selected_sensor, selected_timestamp)
        if selected_sensor in sketch_manager.sensor_model_images:
            plot_images(sketch_manager.sensor_model_images, selected_sensor, selected_timestamp)
        if selected_sensor in sketch_manager.sensor_image_images:
            plot_images(sketch_manager.sensor_image_images, selected_sensor, selected_timestamp)


def handle_model_stats(sketch_manager):
    st.markdown("<h2 style='color: #6C63FF;'>Model Statistics</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #4B4B4B;'>Model Statistics histogram of class accurcay over the classes. In this case there are only 2 classes. Class 0 - Dog and Class 1 - Cat.
    In this case the model in most of the cases certain about its prediction that it either less confident or highly confident, which is good</p>
    <hr style='border-top: 2px solid #6C63FF;'>
    """, unsafe_allow_html=True)
    time_series_stats = sketch_manager.model_stats
    timestamps = list(time_series_stats.keys())
    if len(timestamps) > 1:
        selected_timestamp = st.sidebar.select_slider("Select Timestamp", options=timestamps, value=timestamps[0])
        selected_timestamp_str = datetime.utcfromtimestamp(int(selected_timestamp)).strftime('%Y-%m-%dT%H:%M:%SZ')  # Convert selected timestamp for display
    elif len(timestamps) == 1:
        selected_timestamp = timestamps[0]
        selected_timestamp_str = datetime.utcfromtimestamp(int(selected_timestamp)).strftime('%Y-%m-%dT%H:%M:%SZ')  # Convert selected timestamp for display
        st.sidebar.text(f"last updated: {selected_timestamp_str}")
    else:
        st.sidebar.text("No timestamps available")
        return  # Exit the function early if no timestamps are available

    if 'histograms_plotted' not in st.session_state:
        st.session_state.histograms_plotted = False

    # Plot histograms if they haven't been plotted yet or selected_metric has changed
    figures = plot_histograms(time_series_stats, selected_timestamp, None)
    # Display figures in a grid layout
    num_columns = 2  # Number of columns in each row
    rows = [figures[i:i + num_columns] for i in range(0, len(figures), num_columns)]

    for row in rows:
        cols = st.columns(num_columns)
        for col, fig in zip(cols, row):
            col.plotly_chart(fig)

       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LensAI Monitor Streamlit app.")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    args = parser.parse_args()
    main(args.data_dir)
