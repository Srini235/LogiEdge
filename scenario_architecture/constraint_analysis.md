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
Whereas, for the edge AI use case, we just need to send a single message to the cloud, which can be done when the fault is present / warning state is reached, depending on the design.
Hence bandwidth is a very crucial point to analyze.

#### Quantitative Analysis

### Latency
#### Why Edge AI is required here?
Latency is another critical issue here. We have to report the fault within 90 seconds.
Assume that data is sent to cloud and processed every time it receives. The cloud round-trip latency plus based on the amount of data it gets, time will be very huge for consecutive classifications and judgments. This can be optimized with edge AI use case.

#### Quantitative Analysis

### Economics:
#### Why Edge AI is required here?
Economics may sound non-critical here, but we have to assess here in the perspective of the cost of failure.
For example, a single failure will cost lakhs of rupees of money as loss.
₹28 lakh vaccine spoilage event when a refrigeration unit failed undetected on a Nashik–Aurangabad route - this is one such example. 
This proves that edge AI is required for this use case.

#### Quantitative Analysis

### Reliability / Connectivity (in this use case)
#### Why is Edge AI required here?
Connectivity is a huge constraint here. Although it uses a cloud-based GPS tracking system, it still loses connectivity in rural Maharashtra and Andhra Pradesh hill regions.
This proves that edge AI is required here.

#### Quantitative Analysis

### Privacy (in this use case)
FreightBridge's pharmaceutical clients require proof that cargo condition data cannot be accessed by unauthorised third parties. This indicates that if we have cloud data transfer every now and then, or via cellular network / GPS / other protocols, chances are that the cellular device may get hacked / GPS fetched so that the trucks may be stolen / seized / some competitor may use them for their advantage, that the pharmaceutical clients lose trust over FreightBridge if the cargo condition data is compromised.
This proves that edge AI is required here.