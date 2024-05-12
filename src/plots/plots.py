import matplotlib.pylab as plt


def plot_metrics(train_acc, test_acc, train_loss, test_loss, volume, volume_std):
    import numpy as np

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

    epochs = [i for i in range(len(train_acc))]

    # Plotting train and test accuracy on the left subplot
    ax1.plot(train_acc, label='Train Accuracy')
    ax1.plot(test_acc, label='Test Accuracy')
    ax1.set_title('Training and Test Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([-0.1, 1.1])

    # Plotting train and test loss in the center subplot
    ax2.plot(train_loss, label='Train Loss')
    ax2.plot(test_loss, label='Test Loss')
    ax2.set_title('Training and Test Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([-0.1, 2.5])

    
    # Plotting volume on the right subplot
    ax3.plot(epochs, volume, label='Volume',  color='green')
    ax3.fill_between(epochs, np.array(volume) - np.array(volume_std), np.array(volume) + np.array(volume_std), color='green', alpha=0.2)
    ax3.set_title('Volume Over Epochs')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Volume')
    ax3.legend()
    ax3.grid(True)
    min_value = np.min(volume)
    max_value = np.max(volume)
    ax3.set_ylim([min_value - min_value * 0.5, max_value + max_value * 0.5])

    plt.tight_layout()
    plt.show()


def plot_pca(x_pca, mask):
    
    x_pca = x_pca[:-1, :]
    original_sample = x_pca[-1, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot
    ax.scatter(x_pca[mask, 0], x_pca[mask, 1], x_pca[mask, 2], c="blue", label="Counterfactual", marker="*")
    ax.scatter(x_pca[~mask, 0], x_pca[~mask, 1], x_pca[~mask, 2], c="red", label="Factual")
    ax.scatter(original_sample[0], original_sample[1], original_sample[2], c="green", s=50, label="Original Sample")
    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Random Points Inside a 3D Sphere')
    plt.legend()
    # Show the plot
    plt.show()