# Task A1: Constraint Analysis

## Constraint Analysis for LogiEdge for FreightBridge cold-chain deployment:

The analysis is done as per Edge AI deployment - most popular term used here - BLERP.

B - Bandwidth
L - Latency
E - Economics
R - Reliability
P - Privacy

### Bandwidth:
#### Why Edge AI is required here?
Bandwidth is a critical issue. The system, if detects data and transmits via cloud, cloud infrastructure costs may be higher for more number of vehicles.

#### Quantitative Analysis
Given: Failure scenario, 1 degree C rise per minute.
System requirement: Detect and alert within 90 seconds of a fault signature appearing in sensor data.
##### Cloud use case:
Assuming temperature readings at 1 Hz and vibration readings at 500 Hz, and door events as discrete events:
= [(1 count per second * 4 bytes, for temperature)
  + (500 counts per second * 4 bytes, for vibration * 3 axes)]
  * 60 seconds * 60 minutes * 24 hours
= 518745632 bytes
= approx 0.483 GB per truck
for pilot (85 trucks) ==> 41.065 GB, if successful, for 265 vehicles ==> 128.026 GB
This means that per day data itself is exploding in size and time.
Cost incurred = 128.026 GB * 0.1 rupees * 1024 (conversion to MB) = 13110 rupees / day
##### Edge Use Case:
Data calculation per truck per day, assuming 200 updates per day depending on classification and judgment.

= (4 bytes for temperature + 4*3 bytes for vibration + 1 bit for door close status) * 200
= (4 + 12 + 1) * 200
= (17) * 200
= 3400 bytes per truck
for pilot (85 trucks) ==> 289000 bytes, if successful, for 265 vehicles ==> 901000 bytes = approx 8.391216397285461e-4 GB --> 99% reduction in data transfer requirement
Cost incurred = (901000/(1024*1024))MB * 0.1 = 0.0859 rupees / day
Hence, **Edge AI saves bandwidth**.

### Latency
#### Why Edge AI is required here?
Latency is another critical issue here. We have to report the fault within 90 seconds.
Assume that data is sent to cloud and processed every time it receives. The cloud round-trip latency plus based on the amount of data it gets, time will be very huge for consecutive classifications and judgments. This can be optimized with edge AI use case.

#### Quantitative Analysis
##### Cloud Use Case:

The Nashik–Aurangabad route to Pune great-circle distance is approximately 500 km. Light travels through optical fibre at approximately 200,000 km/s (two-thirds the speed of light in vacuum). Minimum one-way propagation: 500 / 200,000 = 0.0025 s = 2.5 ms. Round-trip minimum: 5 ms. Network routing, switching, 4G Radio Access Network (RAN) transmission, and cloud queuing add 90–200 ms of additional overhead in practice, giving approximately 95–205 ms typical RTT.

But catch here is that the Nashik-Aurangabad route has pure rural area and because of it, if the connection gets disrupted and some issue happens, then the data won't be transmitted and inference itself won't happen, hence issues won't be detected on time.
Hence we need edge AI here.
In edge AI use case, at t=50 seconds this will be solved (since for first 30 seconds the data buffering for sliding window inference happens, then step interval of 2 * 10 seconds = 20 seconds happen), and no network issues are here. Hence this is better.

### Economics:
#### Why Edge AI is required here?
Economics may sound non-critical here, but we have to assess here in the perspective of the cost of failure.
For example, a single failure will cost lakhs of rupees of money as loss.
₹28 lakh vaccine spoilage event when a refrigeration unit failed undetected on a Nashik–Aurangabad route - this is one such example. 
This proves that edge AI is required for this use case.

#### Quantitative Analysis

### Reliability / Connectivity (in this use case)
#### Why is Edge AI required here?
Connectivity is a huge constraint here. Although it uses a cloud-based GPS tracking system, it still loses connectivity in rural Maharashtra and Andhra Pradesh hill regions (7 documented locations so far) and if issues are there for 35-90 minutes then the cloud based system fails.
This proves that edge AI is required here.

### Privacy (in this use case)
FreightBridge's pharmaceutical clients require proof that cargo condition data cannot be accessed by unauthorised third parties. This indicates that if we have cloud data transfer every now and then, or via cellular network / GPS / other protocols, chances are that the cellular device may get hacked / GPS fetched so that the trucks may be stolen / seized / some competitor may use them for their advantage, that the pharmaceutical clients lose trust over FreightBridge if the cargo condition data is compromised.
This proves that edge AI is required here.