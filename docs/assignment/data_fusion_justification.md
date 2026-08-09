## Task C3: Data Fusion Justification

In designing the LogiEdge cold-chain monitoring system, I chose to implement **Feature-Level Fusion**. The preprocessing pipeline extracts statistical features from the temperature and vibration sensors independently over a 30-second window. These are then combined into a single 6-value array: `[temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]` before being fed to the inference engine. 

Here is why this approach is the most practical choice for our edge constraints compared to early or late fusion alternatives:

### 1. Why Data-Level (Early) Fusion Doesn't Work
Data-level fusion involves stitching the raw sensor streams together before doing any processing. For our specific setup, that approach is a non-starter for two main reasons:
* **Mismatched Frequencies:** Our temperature sensor ticks at 1 Hz, while the vibration sensor runs at 0.5 Hz. Trying to fuse raw data directly would force us to artificially upsample the vibration data (essentially guessing data points) or downsample the temperature data (losing valuable resolution). 
* **Incompatible Scales:** We are dealing with degrees Celsius and g-forces. Feeding raw, unscaled physical values with totally different baseline ranges and noise profiles directly into a model makes it significantly harder for the network to learn meaningful patterns.

### 2. The Problem with Decision-Level (Late) Fusion
If we used decision-level fusion, we would need to run the temperature and vibration data through two separate TFLite models and then combine their final predictions (e.g., using a voting logic). I ruled this out because:
* **Missing the Bigger Picture:** Compressor failures are multivariate. A critical failure signature might be a slight temperature drift happening *exactly* when the vibration kurtosis spikes. Two isolated models looking at single streams would completely miss this cross-sensor correlation.
* **Wasting Edge Compute:** As proven in the Roofline Analysis, our hardware is compute-bound. Loading and executing two separate models would double our memory footprint and arithmetic workload. This would threaten our strict 10W power budget and make hitting the 90-second SLA much harder.

### 3. Why Feature-Level Fusion is the Sweet Spot
By extracting features first and then fusing them, we effectively solve both the data and hardware challenges:
* **Natural Time Alignment:** The 30-second sliding window neatly packages our asynchronous data (30 temp readings and 15 vib readings) into a single, unified timeframe, bypassing the need to artificially alter the raw data.
* **Capturing Relationships:** Fusing these 6 extracted features into one vector allows a single, lightweight model to learn exactly how temperature and vibration interact during a hardware degradation cycle.
* **Hardware Efficiency:** We drastically reduce our data dimensionality—compressing 45 raw data points down to just 6 highly representative features. This keeps the input tensor tiny, ensuring inference remains well below the hardware's compute ceiling.