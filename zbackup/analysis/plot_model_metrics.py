
# Import libraries
import matplotlib.pyplot as plt
from pathlib import Path


# Define Model Evaluation Metrics
metrics = {
    "Accuracy": 99.88,
    "Macro Precision": 91.58,
    "Macro Recall": 89.67,
    "Macro F1 Score": 90.49
}


# Define Output Path
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    BASE_DIR / "testing" / "model_evaluation_metrics.png"
)


# Prepare Chart Data
names = list(metrics.keys())
scores = list(metrics.values())


# Generate Model Evaluation Chart
plt.figure(figsize=(10, 6))

bars = plt.barh(
    names,
    scores,
    height=0.55
)

plt.xlim(0, 105)

plt.xlabel("Performance (%)")
plt.title(
    "XGBoost Model Evaluation on CICIDS2017 Test Dataset"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.gca().invert_yaxis()


# Display Metric Values
for bar, score in zip(bars, scores):

    plt.text(
        score + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.2f}%",
        va="center",
        fontsize=11
    )


# Save Chart
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Model evaluation chart generated successfully!")
print("Saved to:", OUTPUT_PATH)
