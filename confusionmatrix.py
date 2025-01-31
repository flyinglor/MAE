import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Known mapping
index_to_label = {0: 'AD', 1: 'CN', 2: 'FTD'}

# Replace these with your predictions and labels
predictions = [2, 0, 0, 2, 2, 1, 2, 2, 2, 0, 2, 0, 1, 1, 0, 2, 2, 0, 2, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 0, 2, 2, 2, 2, 1, 1, 2, 1, 0, 0, 1, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 2, 1, 2, 2, 1, 2, 2, 0, 2, 2, 2, 1, 0, 1, 0, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1, 2, 2, 1, 0, 2, 1, 2, 1, 1, 1, 2, 2, 0, 2, 0, 2, 2, 1, 1, 1, 0, 2]
labels = [0, 2, 0, 2, 2, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0, 1, 1, 0, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 0, 0, 2, 2, 1, 0, 1, 1, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 0, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 1, 0, 2, 2, 1, 1, 2, 0, 0]

# Convert indices to labels
predictions = [index_to_label[idx] for idx in predictions]
labels = [index_to_label[idx] for idx in labels]

# Compute the confusion matrix
cm = confusion_matrix(labels, predictions)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Customize plot
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Save the plot
plt.savefig("confusion_matrix.png", dpi=300)
print("Confusion matrix saved as 'confusion_matrix.png'")

