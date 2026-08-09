## LogiEdge MQTT Edge Architecture Overview

This document outlines the publish-subscribe (Pub/Sub) messaging architecture for the LogiEdge cold-chain monitoring system. To meet the strict reliability, power, and compute constraints of edge computing, the system departs from a monolithic script and instead utilizes a decoupled microservice architecture communicating via a local MQTT broker.

### 1. Hierarchical Topic Namespace
A standardized MQTT topic structure is used to route data efficiently across the system. All topics follow the `project/device_id/stream` format, allowing for seamless scaling if multiple trucks are added to the fleet.
* **Raw Sensor Streams:**
  * `logiedge/truck_LE_01/temperature` (Publishes at 1 Hz)
  * `logiedge/truck_LE_01/vibration` (Publishes at 0.5 Hz)
  * `logiedge/truck_LE_01/door` (Publishes discrete security events)
* **Processed Data Streams:**
  * `logiedge/truck_LE_01/features` (Publishes the fused 6-value vector every 10 seconds)

### 2. Microservice Data Flow
The pipeline is divided into four distinct components to ensure fault tolerance and pipeline integrity:

* **1. Data Generation (Sensor Simulator):** Generates realistic physical readings, injecting simulated compressor degradation and temperature drift. It publishes payloads to the broker using Quality of Service (QoS) level 1 (At-Least-Once delivery). This is critical for edge networks, ensuring no telemetry is lost due to transient connection drops.
* **2. Edge Message Broker (Mosquitto):** Acts as the central nervous system for the edge node. Running locally on the device (localhost:1883), it allows the internal microservices to communicate with sub-millisecond latency without requiring a continuous cloud connection.
* **3. Preprocessing Engine (Data Pipeline):** Subscribes to the raw sensor topics. It safely buffers incoming data into a 30-second sliding window, applies a 5-sample moving average filter, and performs feature-level fusion. It then publishes a single array of extracted features to the `features` topic. 
* **4. AI Inference Service:** Subscribes exclusively to the `features` topic. It normalizes the incoming vector, executes the TFLite anomaly detection model, tracks consecutive critical predictions for alerting, and logs the final output securely to a local SQLite database (WAL mode enabled for concurrent access).

### 3. Architectural Justification for Edge Deployment
Decoupling the ingestion pipeline from the AI inference engine provides three major enterprise-grade advantages:
* **Fault Isolation:** If the machine learning runtime crashes, runs out of memory, or undergoes a restart, the preprocessing service remains entirely unaffected. It will continue to buffer sensor data and maintain the integrity of the sliding window. 
* **Seamless OTA Updates:** In a production setting, deploying retrained TFLite models via Over-The-Air (OTA) updates is standard practice. Because the services are split, the AI container can be swapped out and restarted without taking the mission-critical sensor ingestion offline.
* **CPU Resource Allocation:** Splitting the workload prevents the heavy arithmetic of the neural network from blocking the thread responsible for capturing the strict 1 Hz and 0.5 Hz sensor ticks, preventing loop drift and data loss.