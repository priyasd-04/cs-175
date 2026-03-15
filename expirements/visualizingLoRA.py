from torch import tensor
import matplotlib.pyplot as plt

results = {0: (0.27323578262612935, tensor(2.436802625656128)), 10: (0.3174882580836614, tensor(2.1763)), 25: (0.3737682778210867, tensor(2.1835)), 50: (0.3410178933824812, tensor(2.1748)), 75: (0.392479528983434, tensor(2.1820)), 100: (0.3555063786251204, tensor(2.1831))}

x = sorted([key for key in results.keys()])
shakespeare_likeliness = [results[key][0] for key in x]
loss = [float(results[key][1]) for key in x]

fig, axs = plt.subplots(1, 2)
axs[0].plot(x, shakespeare_likeliness, label="Judge Score")
axs[0].set_ylim(0,1)
axs[0].set_title('Shakespeare Likeliness vs. Data %', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Proportion of Training Data (%)')
axs[0].set_ylabel('Probability Score $[0, 1]$')

axs[1].plot(x, loss, label="Cross Entropy Loss")
axs[1].set_ylim(0,3)
axs[1].set_title('Test Loss vs. Data %', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Proportion of Training Data (%)')
axs[1].set_ylabel('Loss Value')


plt.suptitle('LoRA Fine-Tuning Performance Analysis', fontsize=14)
plt.tight_layout()
plt.show()
