# Hardware Justification

## Constraint Analysis for the Hardware Justifications to be done

### Option 1 - Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L):

#### Compute Performance
The 13 TOPS Hailo-8L accelerator effortlessly processes the TFLite models on the 30-second sliding windows, consistently meeting the 90-second latency SLA. 
It also supports the full Linux/Docker stack required for the Mosquitto broker and SQLite database designed in Component A.

#### Power Budget
1. Power budget which could be provided by Truck = 10W AI power budget (12V truck supply via DC-DC converter)
2. TDP from Raspberry Pi 5 8 GB + AI HAT+ = 7.5 W (as given in question). Sometimes in peak load it may rise to 9.5 W, still within the power budget.

#### Unit Cost
1. Unit Cost budget per device in truck = ~₹15,000/truck (India Price)
For a complete fleet = 85 trucks pilot, 265 at full scale 
= 15000 * 265 = Rs 39,75,000 (for full scale)
2. CapEx Costs comparison:
Truck CapEx: ~₹40,00,000 per truck
Edge Node (Pi 5) CapEx: ₹15,000 per truck
Ratio: Rs 15,000 / 40,00,000 = 0.00375 = 0.38%
The Raspberry Pi 5 setup costs less than 0.38% of the vehicle's base cost.
Given that a single cargo spoilage event costs ₹28,00,000, this ₹15,000 investment acts as highly cost-effective insurance, paying for itself by preventing just a single failure.

### Option 2 - Jetson Orin Nano Super Developer Kit (67 TOPS):

#### Compute Performance
The 67 TOPS provided is vastly over-provisioned for a simple 3-class TFLite model running on a 30-second window, meaning FreightBridge would be paying for compute they will never use. Hence it is a performance overkill.

#### Power Budget
1. Power budget which could be provided by Truck = 10W AI power budget (12V truck supply via DC-DC converter)
2. TDP from Jetson Orin Nano Super Developer Kit (67 TOPS) = 15W moderate load(as given in question). Sometimes in peak load it may rise to 25 W.
Either way, the TDP > Power budget here, hence it is not a viable option.

#### Unit Cost
1. Unit Cost budget per device in truck = ~₹45,000/truck (India Price)
For a complete fleet = 85 trucks pilot, 265 at full scale 
= 45000 * 265 = ₹1,19,25,000 (for full scale)
2. CapEx Costs comparison:
Truck CapEx: ~₹40,00,000 per truck
Edge Node (Jetson Orin Nano) CapEx: ₹45,000 per truck
Ratio: 45,000 / 40,00,000 = 0.01125 = 1.13%
The Jetson Orin Nano Super Developer Kit setup costs less than 1.13% of the vehicle's base cost.   
Given the ₹28,00,000 cost of a single cargo spoilage event, spending ₹1.19 Crores across the fleet on over-provisioned hardware severely diminishes the return on investment.


### Option 3 - STM32H7-based custom MCU with sensor ICs:

#### Compute Performance
It cannot run the Linux OS, Docker containers, Python preprocessing scripts, SQLite WAL, or the local Mosquitto broker outlined in the system architecture.
Hence it is a complete failure in compute performance and compatibility point of view.

#### Power Budget
1. Power budget which could be provided by Truck = 10W AI power budget (12V truck supply via DC-DC converter)
2. TDP from STM32H7-based custom MCU with sensor ICs = 0.4W (as given in question).

Comparison result: The amount of power consumed by the MCU is very less, hence this seems the most viable option, in power budget perspective.

#### Unit Cost
1. Unit Cost budget per device in truck = ~₹3,500/truck (India Price)
For a complete fleet = 85 trucks pilot, 265 at full scale 
= 3500 * 265 = Rs 9,27,500 (for full scale)
2. CapEx Costs comparison:
Truck CapEx: ~₹40,00,000 per truck
Edge Node (STM32 setup) CapEx: ₹3,500 per truck
Ratio: 3,500 / 40,00,000 = 0.000875 = 0.09%
The STM32 setup costs less than 0.09% of the vehicle's base cost.   
Given the ₹28,00,000 cost of a single cargo spoilage event, choosing this solely because it is the most cost-effective hardware creates a false economy, as its software limitations risk missing the failures entirely.

## Comparison Table:

| Metric / Hardware | Option 1: Raspberry Pi 5 + AI HAT+ | Option 2: Jetson Orin Nano Dev Kit | Option 3: STM32H7 Custom MCU |
| :--- | :--- | :--- | :--- |
| **Compute / Architecture** | 13 TOPS (Hailo-8L); Runs Linux + Docker | 67 TOPS; Runs Linux + Docker | No MMU; Bare-metal C / FreeRTOS only |
| **Latency SLA Capacity** | **Pass** (Processes 30s window in ~20 ms) | **Pass** (Over-provisioned overkill) | **Fail** (SRAM limits high-freq buffer) |
| **Power Consumption (TDP)**| **7.5W** (Fits <10W budget safely) | **15W** (Violates 10W vehicle ceiling) | **0.4W** (Highly efficient) |
| **Unit Cost (India Price)** | ₹15,000 | ₹45,000 | ₹3,500 |
| **CapEx Ratio (per ₹40L Truck)**| **0.38%** | 1.13% | 0.09% |
| **Pilot Phase Cost (85 units)**| **₹12,75,000** | ₹38,25,000 | ₹2,97,500 |
| **Full Scale Cost (265 units)**| **₹39,75,000** | ₹1,19,25,000 | ₹9,27,500 |
| **Constraint Fit Evaluation**| **Optimal Fit** | **Power & Cost Violator** | **Compute & Architecture Violator** |

## Conclusion - Selection of Hardware and Justification
Ultimately, the truck's strict 10W power limit makes power the deciding constraint, immediately ruling out the 15W Jetson. And while the STM32 is incredibly cheap and efficient, it simply cannot run the required Linux software stack. This makes the **Raspberry Pi 5 + AI HAT+** the only practical choice that guarantees the 90-second safety SLA while satisfying both the budget and the electrical constraints.