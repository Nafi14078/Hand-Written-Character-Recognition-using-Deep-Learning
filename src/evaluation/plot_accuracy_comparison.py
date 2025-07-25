import matplotlib.pyplot as plt

# Labels and values
labels = ['MNIST Test Accuracy', 'Custom Data Accuracy']
accuracies = [99.07, 60.0]

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=['green', 'orange'])
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.title('Model Accuracy Comparison')

# Annotate bar values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=12)

# Save and show
plt.tight_layout()
plt.savefig('outputs/prediction_visuals/accuracy_comparison.png')
plt.show()
