import numpy as np

# Data extraction
test_loss = np.array([0.88422, np.nan, np.nan, 1.17066, np.nan])
accuracy = np.array([0.66292, 0.25843, 0.25843, 0.39326, 0.25843])
balanced_accuracy = np.array([0.65343, 0.33333, 0.33333, 0.33333, 0.33333])
precision = np.array([0.64019, 0.08614, 0.08614, 0.13109, 0.08614])
recall = np.array([0.65343, 0.33333, 0.33333, 0.33333, 0.33333])
f1_score = np.array([0.63783, 0.13690, 0.13690, 0.18817, 0.13690])


# Calculation of mean and standard deviation
metrics = {
    "Test loss": test_loss,
    "Accuracy": accuracy,
    "Balanced Accuracy": balanced_accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1_score
}

results = {}
for metric, values in metrics.items():
    results[metric] = {
        "mean": np.mean(values),
        "std": np.std(values)  # Sample standard deviation
    }

# Print results
for metric, stats in results.items():
    print(f"{metric}:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}\n")