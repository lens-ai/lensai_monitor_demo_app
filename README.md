# Lens AI Monitoring Demo App
This demo monitoring app showcases sample monitoring metrics and observability KPIs. The app is developed using streamlit
Please explore the app: https://lensai-demo-app.streamlit.app

## Features

- Display image data metrics
- Display model metrics
- Display sensor level metrics and sampled data

## Installation

1. Clone the repository:
```sh
git clone https://github.com/lens-ai/lensai_monitor_demo_app.git
cd lensai_monitor_demo_app
```

2. Create a virtual env
```sh
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

3. Run the app
```sh
streamlit run lensai_monitor_app.py
```

## Data directory format
The app expects the data directory structure as follows
![Directory](directoryPath.png)
sensor2
### for more documentation refer: https://app.gitbook.com/o/QHDiYcZjAf0R6mzJPPHn/s/aBxpe4kLU8KyHMlQtPS1/~/changes/6/fundamentals/getting-started/step-3-monitoring-the-metrics-on-the-lens-ai-server.
