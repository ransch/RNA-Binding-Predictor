import pickle
import matplotlib.pyplot as plt
import sys


def main():
    history_file_path = sys.argv[1]
    with open(history_file_path, 'rb') as f:
        history = pickle.load(f)
    print("min val loss:", min(history['val_loss']))
    print("max val acc:", max(history['val_accuracy']))
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
