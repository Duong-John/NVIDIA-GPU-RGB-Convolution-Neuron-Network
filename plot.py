import json
import matplotlib.pyplot as plt

with open('report/log.json', 'r') as f:
    data = json.load(f)

epochs = [item['epoch'] for item in data]
t_loss = [item['t_loss'] for item in data]
v_loss = [item['v_loss'] for item in data]
t_acc  = [item['t_acc'] for item in data]
v_acc  = [item['v_acc'] for item in data]
time_ms = [item['time_ms'] for item in data]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(epochs, t_loss, label='Train Loss', color='blue', linewidth=2)
ax1.plot(epochs, v_loss, label='Validation Loss', color='orange', linewidth=2)
ax1.set_title('Training vs Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, t_acc, label='Train Accuracy', color='green', linewidth=2)
ax2.plot(epochs, v_acc, label='Validation Accuracy', color='red', linewidth=2)
ax2.set_title('Training vs Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

ax3.plot(epochs, time_ms, label='Time per Epoch (ms)', color='purple', linewidth=2)
ax3.set_title('Execution Time per Epoch')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Time (milliseconds)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()

plt.show()