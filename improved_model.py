import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def create_improved_model(input_shape, num_classes):
    # Input layer
    model_input = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(model_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Second convolutional block
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Third convolutional block
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    model_output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=model_input, outputs=model_output)
    
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    # Create model
    model = create_improved_model(input_shape, num_classes)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
    
    # Compile model with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

def preprocess_data(X):
    # Normalize the data
    scaler = StandardScaler()
    X_reshaped = X.reshape(X.shape[0], -1)
    X_normalized = scaler.fit_transform(X_reshaped)
    return X_normalized.reshape(X.shape)

# Example usage:
"""
# After loading your data:
X = preprocess_data(X)
y_encoded = to_categorical(y, NUM_CLASSES)

# Split data
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y_encoded[:split_index], y_encoded[split_index:]

# Train model
input_shape = (N_MELS, X_train.shape[2], 1)
model, history = train_model(X_train, y_train, X_val, y_val, input_shape, NUM_CLASSES)
""" 