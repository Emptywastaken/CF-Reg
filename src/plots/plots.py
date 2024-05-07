import matplotlib.pylab as plt


def plot_metrics(train_acc, test_acc, train_loss, test_loss, volume, volume_std):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

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
    ax3.errorbar(epochs, volume, label='Volume', yerr=volume_std, color='green')
    ax3.set_title('Volume Over Epochs')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Volume')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim([-5, 130])

    plt.tight_layout()
    plt.show()