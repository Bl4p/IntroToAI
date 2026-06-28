import matplotlib.pyplot as plt
import numpy as np
import os

# Data
models = ['TG-GRU', 'Isolation Forest', 'Transformer']
roc_auc = [0.036, 0.856, 0.522]
pr_auc = [0.089, 0.386, 0.365]
eer = [0.879, 0.194, 0.512]
latency = [0.173, 6.176, 0.360]

x = np.arange(len(models))
width = 0.25

# 1. Accuracy Metrics Chart
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, roc_auc, width, label='ROC-AUC (Higher is better)', color='#5D9CEC')
rects2 = ax.bar(x, pr_auc, width, label='PR-AUC (Higher is better)', color='#48CFAD')
rects3 = ax.bar(x + width, eer, width, label='EER (Lower is better)', color='#ED5565')

ax.set_ylabel('Score')
ax.set_title('Classification Accuracy Metrics (ROC-AUC, PR-AUC, EER)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

fig.tight_layout()
plt.savefig('Ver2/accuracy_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Latency Chart
fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, latency, 0.5, label='Mean Latency (ms)', color=['#AC92EC', '#FFCE54', '#656D78'])

ax.set_ylabel('Milliseconds (ms)')
ax.set_title('Inference Latency (ms)')
ax.set_xticks(x)
ax.set_xticklabels(models)

fig.tight_layout()
plt.savefig('Ver2/latency_chart.png', dpi=300, bbox_inches='tight')
plt.close()
