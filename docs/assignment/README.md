# LogiEdge — ML on the Edge Assignment Group 13

This repository contains the LogiEdge cold-chain monitoring solution developed for the MTech AI/ML WILP assignment. LogiEdge is designed to run edge-aware anomaly detection and drift monitoring for refrigerated freight transport, using a lightweight MQTT-based microservice pipeline.

The project is hardware-agnostic by design and supports multiple deployment environments:

- Raspberry Pi 5 (tested with Ansible playbook and individual scripts)
- Jetson Xavier NX running Ubuntu 20.04 (tested with Ansible playbook and individual scripts)
- PC/WSL (individual scripts work; Ansible deployment has not been tested)

## What this repository includes

- `data_pipeline/` — sensor simulation, preprocessing, and MQTT integration
- `inference/` — edge inference service and TFLite model artifacts
- `monitoring/` — drift monitoring and PSI alerting
- `deployment/` — Ansible playbook and service deployment manifests
- `optimization/` — benchmarks, models, and deployment recommendation
- `hardware/`, `scenario_architecture/`, `problem_statement/`, `report/` — support documents and analysis

## LogiEdge problem and solution summary

LogiEdge solves cold-chain cargo monitoring for freight transport by moving anomaly detection to the edge. The system detects refrigeration failures, temperature drift, and vibration anomalies locally, then issues alerts without relying on constant cloud connectivity.

This is important because:

- rural routes often suffer cellular outages
- raw sensor streaming to cloud is expensive and slow
- regulatory and client requirements demand privacy and fast reaction times

The solution uses a local MQTT broker, feature-level fusion, TFLite inference, and drift monitoring to ensure the system is reliable, low-bandwidth, and suitable for on-vehicle deployment.

## Raspberry Pi setup and execution

These steps were tested on Raspberry Pi 5 after required libraries were installed.

### Step 1: Install system dependencies & MQTT broker

The microservices rely on an MQTT broker to pass messages locally. Update your system and install Mosquitto, Ansible, Git, Python tools, and the MQTT clients.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install mosquitto mosquitto-clients ansible git python3-pip python3-venv -y
sudo systemctl enable --now mosquitto
```

### Step 2: Clone the repository

Clone the project repository directly from GitHub onto the device.

```bash
cd ~
git clone https://github.com/Srini235/MLontheEdge_Assignment_Group_13.git
cd MLontheEdge_Assignment_Group_13
```

### Step 3: Set up the Python virtual environment

Create and activate a local Python virtual environment, then install the required packages.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy paho-mqtt tensorflow
```

### Step 4: Deploy the background services via Ansible

Automate the configuration of the preprocessor and monitor daemons using the deployment playbook.

```bash
cd deployment
ansible-playbook logiedge_deploy.yml -i localhost, -c local --ask-become-pass
```

Enter your device sudo password when prompted.

Verify that the background systemd services are active and running:

```bash
sudo systemctl status logiedge-preprocessor.service
sudo systemctl status logiedge-monitor.service
```

### Step 5: Execute and test the live pipeline

Open two separate terminal windows to run the live validation.

#### Terminal 1: Watch the drift monitor logs

Stream real-time output from the drift monitor service to observe PSI calculations:

```bash
sudo journalctl -u logiedge-monitor.service -f
```

#### Terminal 2: Run the simulator

Navigate to the data pipeline, activate the virtual environment, and inject telemetry streams.

```bash
cd ~/MLontheEdge_Assignment_Group_13/data_pipeline
source ../venv/bin/activate

# 1. Inject normal behavior (Class 0 baseline)
python3 simulator.py --anomaly none

# 2. After observing normal status, press Ctrl+C, then inject anomalies:
python3 simulator.py --anomaly temp_drift
```

Watch Terminal 1 update in real time as the background inference service evaluates the model and the drift monitor calculates the Population Stability Index (PSI).

## Supported environments

- **Raspberry Pi 5** — tested with Ansible deployment and individual scripts
- **Jetson Xavier NX (Ubuntu 20.04)** — tested with Ansible deployment and individual scripts
- **PC / WSL** — individual scripts work; Ansible deployment is not tested on this platform

## Documentation

The repository includes a docs-as-code site for easy browsing. Use the following links from the repository root:

```bash
cd docs
mkdocs serve
```

Then open the local preview page shown by MkDocs.

- [Docs landing page](docs/index.md)
- [Assignment documentation index](docs/assignment/README.md)

## Notes

- The repository contains additional documentation in `docs/` and the assignment analysis in `hardware/`, `scenario_architecture/`, `problem_statement/`, and `report/`.
- The LogiEdge architecture is designed to be hardware-agnostic, with all core processing implemented using Python, MQTT, and TensorFlow Lite.
