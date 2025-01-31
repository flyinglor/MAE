import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "mask_ratio": [75, 80, 85, 90],
    "DZNE": [
        (0.6313, 0.0697),
        (0.6458, 0.0942),
        (0.6210, 0.0630),
        (0.3974, 0.1280),
    ],
    "HOS": [
        (0.6727, 0.0314),
        (0.6469, 0.0287),
        (0.5125, 0.0368),
        (0.4948, 0.1434),
    ],
}

# Extract mask ratios and values with errors
mask_ratios = data["mask_ratio"]

# DZNE values and errors
dzne_values = [val[0] for val in data["DZNE"]]
dzne_errors = [val[1] for val in data["DZNE"]]

# HOS values and errors
hos_values = [val[0] for val in data["HOS"]]
hos_errors = [val[1] for val in data["HOS"]]

# Plotting
plt.figure(figsize=(8, 6))

# DZNE plot
plt.errorbar(
    mask_ratios,
    dzne_values,
    yerr=dzne_errors,
    label="DZNE Dataset",
    fmt="o-",
    capsize=5,
    color="lightcoral",
)

# HOS plot
plt.errorbar(
    mask_ratios,
    hos_values,
    yerr=hos_errors,
    label="In-hospital Dataset",
    fmt="s-",
    capsize=5,
    color="yellowgreen",
)

# Customize plot
plt.title("Performance vs Mask Ratio")
plt.xlabel("Mask Ratio (%)")
plt.ylabel("Balanced Accuracy")
plt.xticks(mask_ratios)
plt.legend()
plt.grid(alpha=0.3)

# Save plot
plt.tight_layout()
plt.savefig("performance_vs_mask_ratio.png")
plt.show()
