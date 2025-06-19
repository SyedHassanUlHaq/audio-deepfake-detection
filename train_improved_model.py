import subprocess
import sys
import os
import numpy as np
import librosa
import tensorflow as tf
from improved_model import create_improved_model, train_model, preprocess_data
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import glob

# --- Auto-discover label file and data directory ---
def find_label_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "ASVspoof2019.LA.cm.train.trn.txt":
                return os.path.join(dirpath, filename)
    return None

def find_data_dir(protocol_dir):
    for dirpath, dirnames, filenames in os.walk(protocol_dir):
        if any(f.endswith('.flac') for f in filenames):
            return dirpath
    return None

# Set kagglehub cache path
kagglehub_cache = os.path.expanduser("~/.cache/kagglehub/datasets/awsaf49/asvpoof-2019-dataset/versions/1")

LABEL_FILE_PATH = find_label_file(kagglehub_cache)
DATASET_PATH = None

if LABEL_FILE_PATH:
    # Get the protocol directory (should be .../LA/LA)
    protocol_dir = os.path.dirname(os.path.dirname(LABEL_FILE_PATH))
    # Prefer the train set directory
    train_flac_dir = os.path.join(protocol_dir, "ASVspoof2019_LA_train", "flac")
    if os.path.isdir(train_flac_dir):
        DATASET_PATH = train_flac_dir
    else:
        DATASET_PATH = find_data_dir(protocol_dir)

if LABEL_FILE_PATH and DATASET_PATH:
    print("Dataset already present. Skipping download.")
    print(f"Using label file: {LABEL_FILE_PATH}")
    print(f"Using data directory: {DATASET_PATH}")
else:
    # Ensure kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub'])
        import kagglehub
    # Download latest version of the dataset
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
    print("Path to dataset files:", path)
    LABEL_FILE_PATH = find_label_file(path)
    protocol_dir = os.path.dirname(os.path.dirname(LABEL_FILE_PATH))
    DATASET_PATH = find_data_dir(protocol_dir)
    if not LABEL_FILE_PATH or not DATASET_PATH:
        raise FileNotFoundError("Could not find required label file or .flac data directory after download.")
    print(f"Using label file: {LABEL_FILE_PATH}")
    print(f"Using data directory: {DATASET_PATH}")

NUM_CLASSES = 2  # Number of classes (bonafide and spoof)
SAMPLE_RATE = 16000  # Sample rate of your audio files
DURATION = 5  # Duration of audio clips in seconds
N_MELS = 128  # Number of Mel frequency bins

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    labels = {}

    with open(LABEL_FILE_PATH, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label = 1 if parts[-1] == "bonafide" else 0
        labels[file_name] = label

    X = []
    y = []

    max_time_steps = 109  # Define the maximum time steps for your model

    for file_name, label in labels.items():
        file_path = os.path.join(DATASET_PATH, file_name + ".flac")
        if not os.path.isfile(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        # Load audio file using librosa
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        # Extract Mel spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Ensure all spectrograms have the same width (time steps)
        if mel_spectrogram.shape[1] < max_time_steps:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :max_time_steps]
        X.append(mel_spectrogram)
        y.append(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Preprocess the data
    X = preprocess_data(X)
    y_encoded = to_categorical(y, NUM_CLASSES)

    # Split the data
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y_encoded[:split_index], y_encoded[split_index:]

    return X_train, X_val, y_train, y_val

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    
    # Define input shape
    input_shape = (N_MELS, X_train.shape[2], 1)
    
    print("Training model...")
    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, NUM_CLASSES)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save('improved_audio_classifier.h5')
    print("Model saved as 'improved_audio_classifier.h5'")
    
    # Print final validation accuracy
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")

if __name__ == "__main__":
    main() 