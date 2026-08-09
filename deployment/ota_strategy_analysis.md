# Task E3 — OTA Strategy Selection Analysis

## Scenario Overview

- **Model update frequency:** Every 6 weeks
- **Model file size:** 280 KB (INT8 TFLite)
- **Connectivity:** M2M SIM at ₹0.10/MB
- **Fleet size:** 85 trucks in pilot
- **Update cycle cost calculation:** Per 6-week period

---

## Strategy 1: Full Fleet Replacement

### Description
Deploy the new model to all 85 trucks simultaneously.

### Bandwidth Calculation
```
Per-truck download: 280 KB
Total fleet: 85 trucks × 280 KB = 23,800 KB = 23.8 MB
Cost: 23.8 MB × ₹0.10/MB = ₹2.38 per update cycle
```

### Annual Cost (52 weeks / 6 = ~8.67 cycles)
```
₹2.38 × 8.67 = ₹20.63 per year
```

### Pros
- Simplest to implement
- Fastest full deployment
- Uniform model across fleet

### Cons
- No rollback capability
- No early warning if model has issues
- All trucks affected by any bugs

---

## Strategy 2: Canary Release (10 trucks first)

### Description
Deploy to 10 trucks first (11.8% of fleet), monitor for 1 week, then roll out to remaining 75 trucks.

### Bandwidth Calculation
```
Canary batch: 10 trucks × 280 KB = 2,800 KB = 2.8 MB
Full rollout: 75 trucks × 280 KB = 21,000 KB = 21 MB
Total per cycle: 2.8 + 21 = 23.8 MB (same as full, but staged)
```

### Cost
```
Cost: 23.8 MB × ₹0.10/MB = ₹2.38 per update cycle
```

### Pros
- Limited exposure to bugs (only 10 trucks)
- Real-world validation before full rollout
- Easy rollback (just don't proceed to full rollout)

### Cons
- Higher operational complexity
- Same total bandwidth (staggered, not saved)
- Requires monitoring infrastructure

---

## Strategy 3: Shadow Mode

### Description
Run new model alongside existing model on edge (shadow), compare outputs, don't switch until validated.

### Bandwidth Calculation
```
Shadow requires downloading new model but NOT switching:
Per-truck shadow download: 280 KB
Total: 85 trucks × 280 KB = 23.8 MB
Cost: 23.8 MB × ₹0.10/MB = ₹2.38 per update cycle
```

### Additional Considerations
- Requires 2x storage on edge device
- Inference cost doubles (run both models)
- Adds latency overhead

### Pros
- Zero risk to operations
- A/B testing in production
- Maximum safety

### Cons
- Highest compute cost (2x inference)
- Same bandwidth as full replacement
- Not suitable for safety-critical cold-chain (delays matter)

---

## Bandwidth Cost Comparison

| Strategy | Per Cycle | Annual (8.67 cycles) |
|----------|-----------|---------------------|
| Full Replacement | 23.8 MB | ₹20.63 |
| Canary (10 trucks) | 23.8 MB | ₹20.63 |
| Shadow Mode | 23.8 MB | ₹20.63 |

**Note:** All strategies have identical bandwidth because the model file must reach all trucks. The difference is *when* and *with what safeguards*.

---

## Recommendation: Canary Release (10 trucks)

### Rationale for Cold-Chain Safety-Critical Systems

1. **Cold-Chain Safety Criticality**
   - Incorrect temperature predictions can lead to cargo spoilage worth lakhs
   - Regulatory compliance requires documented validation
   - Cannot risk entire fleet on unvalidated model

2. **Rural Connectivity Characteristics**
   - M2M SIM in remote areas has variable latency (2-15 seconds per MB)
   - Spotty connectivity means OTA may fail on some trucks
   - Staggered rollout handles intermittent connectivity better

3. **Operational Risk Mitigation**
   - 10-truck canary = 11.8% exposure vs 100%
   - 1-week monitoring window catches regressions
   - Clear go/no-go decision criteria before full rollout

4. **Cost-Effective Safety**
   - Same bandwidth cost as full replacement (₹2.38/cycle)
   - Dramatically reduces blast radius of bugs
   - Enables rapid rollback without truck-side intervention

### Why NOT the others:

**Full Replacement:**
- Unacceptable risk for safety-critical cargo
- No way to detect bugs before all 85 trucks affected
- Would require emergency rollback procedure across entire fleet

**Shadow Mode:**
- Doubles inference compute (costly on edge)
- Not suitable for latency-sensitive operations
- Cold-chain decisions need to be fast, not "wait for validation"
- Storage constraints on edge devices

---

## Summary

| Factor | Full | Canary | Shadow | Winner |
|--------|------|--------|--------|--------|
| Bandwidth | 23.8 MB | 23.8 MB | 23.8 MB | Tie |
| Safety Risk | High | Low | Low | Canary |
| Rollback Complexity | Hard | Easy | Easy | Canary |
| Operational Overhead | Low | Medium | High | Full |
| Cold-Chain Suitability | No | Yes | Partial | **Canary** |

**Final Recommendation:** Deploy via **Canary Release** with 10 trucks for 1 week monitoring before full fleet rollout.
