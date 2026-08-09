"""
pareto_analysis_logiedge.py
Computes and visualizes the Pareto frontier for the LogiEdge tabular models.
"""
import numpy as np
import matplotlib.pyplot as plt

# ── LOGIEDGE BENCHMARK MEASUREMENTS ─────────────
RESULTS = {
    'M1 (FP32 Baseline)': {
        'size_kb':    5.28,
        'latency_ms': 0.0167,
        'accuracy':   100.00,
        'energy_mj':  0.0001,
    },
    'M2 (PTQ INT8)': {
        'size_kb':    4.92,
        'latency_ms': 0.0174,
        'accuracy':   99.49,
        'energy_mj':  0.0000,
    },
    'M3 (Pruned + INT8)': {
        'size_kb':    4.92,
        'latency_ms': 0.0158,
        'accuracy':   99.49,
        'energy_mj':  0.0000,
    }
}

# ── Summary table ──────────────────────────────────────────
print('\n' + '='*80)
print(f' {"Model":<25} {"Size (KB)":>10} {"Latency (ms)":>14} {"Acc (%)":>10} {"Energy (mJ)":>12}')
print('='*80)
for name, m in RESULTS.items():
    print(f' {name:<25} {m["size_kb"]:>10.2f} {m["latency_ms"]:>14.4f} '
          f'{m["accuracy"]:>10.2f} {m["energy_mj"]:>12.4f}')
print('='*80)

# ── Extract arrays for computation ─────────────────────────
names     = list(RESULTS.keys())
latencies = np.array([RESULTS[n]['latency_ms'] for n in names])
accs      = np.array([RESULTS[n]['accuracy']   for n in names])
sizes     = np.array([RESULTS[n]['size_kb']    for n in names])

def pareto_frontier(x_lower_better, y_higher_better):
    """Return indices of Pareto-optimal points (lower x, higher y)."""
    n = len(x_lower_better)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j is better or equal on BOTH dimensions, 
                # and strictly better on at least one.
                if (x_lower_better[j] <= x_lower_better[i] and
                    y_higher_better[j] >= y_higher_better[i] and
                    (x_lower_better[j] < x_lower_better[i] or
                     y_higher_better[j] > y_higher_better[i])):
                    is_pareto[i] = False
                    break
    return is_pareto

pareto_mask = pareto_frontier(latencies, accs)

print('\n Pareto-optimal models (Accuracy vs Latency):')
for i, name in enumerate(names):
    status = '★ Pareto-optimal' if pareto_mask[i] else '  Dominated     '
    print(f'   {status}  →  {name}')

# ── Visualization ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colours = ['#E07B39', '#1C7293', '#02C39A']

# Plot 1: Accuracy vs Latency
ax1 = axes[0]
for i, name in enumerate(names):
    marker = '*' if pareto_mask[i] else 'o'
    size   = 300 if pareto_mask[i] else 100
    ax1.scatter(latencies[i], accs[i], s=size, color=colours[i],
                zorder=5, marker=marker, label=name)
    
    # Adjust text offset to prevent overlapping
    offset_y = 0.2 if i == 2 else -0.4
    ax1.annotate(name, (latencies[i], accs[i]),
                 textcoords='offset points', xytext=(6, offset_y * 10), fontsize=10)

ax1.set_xlabel('Mean Inference Latency (ms)  ← lower is better', weight='bold')
ax1.set_ylabel('Top-1 Accuracy (%)  ↑ higher is better', weight='bold')
ax1.set_title('Pareto Frontier: Accuracy vs Latency', weight='bold')
# Add padding to axes to ensure points aren't cut off
ax1.set_xlim(min(latencies) - 0.0005, max(latencies) + 0.001)
ax1.set_ylim(min(accs) - 0.5, max(accs) + 0.5)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(alpha=0.3, linestyle='--')


# Plot 2: Accuracy vs Model Size
ax2 = axes[1]
pareto2 = pareto_frontier(sizes, accs)

for i, name in enumerate(names):
    marker = '*' if pareto2[i] else 'o'
    size2  = 300 if pareto2[i] else 100
    ax2.scatter(sizes[i], accs[i], s=size2, color=colours[i],
                zorder=5, marker=marker, label=name)
    
    # Adjust text offset
    offset_y = 0.2 if i == 2 else -0.4
    ax2.annotate(name, (sizes[i], accs[i]),
                 textcoords='offset points', xytext=(6, offset_y * 10), fontsize=10)

ax2.set_xlabel('Model Size (KB)  ← smaller is better', weight='bold')
ax2.set_ylabel('Top-1 Accuracy (%)  ↑ higher is better', weight='bold')
ax2.set_title('Pareto Frontier: Accuracy vs Model Size', weight='bold')
ax2.set_xlim(min(sizes) - 0.2, max(sizes) + 0.3)
ax2.set_ylim(min(accs) - 0.5, max(accs) + 0.5)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('logiedge_pareto_analysis.png', dpi=200, bbox_inches='tight')
print('\n Pareto plot saved to logiedge_pareto_analysis.png')
#plt.show() # Uncomment to display the plot interactively