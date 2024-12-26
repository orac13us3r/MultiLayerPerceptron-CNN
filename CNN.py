import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, Activation)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

def build_cnn(input_shape=(32, 32, 3), kernel_size=(3, 3), learning_rate=0.001, batch_norm=True):
    model = Sequential()

    # First Conv Layer
    model.add(Conv2D(32, kernel_size, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Second Conv Layer
    model.add(Conv2D(32, kernel_size, padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Max Pooling + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Third Conv Layer
    model.add(Conv2D(64, kernel_size, padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Fourth Conv Layer
    model.add(Conv2D(64, kernel_size, padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Max Pooling + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Fifth Conv Layer
    model.add(Conv2D(128, kernel_size, padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Sixth Conv Layer
    model.add(Conv2D(128, kernel_size, padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())

    # Max Pooling + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, test_loss, test_accuracy

def plot_history(history, title):
    plt.figure(figsize=(10, 6))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def experiment():
    x_train, y_train, x_test, y_test = load_cifar10()

    # Original model settings
    print("\nOriginal CNN with default parameters")
    model = build_cnn()
    history, _, test_accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)
    plot_history(history, 'Original Model')
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Experiment 1: Change learning rate
    for lr in [0.05, 0.0001]:
        print(f"\nExperiment: Learning Rate = {lr}")
        model = build_cnn(learning_rate=lr)
        history, _, test_accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)
        plot_history(history, f'Learning Rate = {lr}')
        print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Experiment 2: Change kernel size
    print("\nExperiment: Kernel Size = 7x7")
    model = build_cnn(kernel_size=(7, 7))
    history, _, test_accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)
    plot_history(history, 'Kernel Size = 7x7')
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Experiment 3: Remove batch normalization
    print("\nExperiment: Without Batch Normalization")
    model = build_cnn(batch_norm=False)
    history, _, test_accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)
    plot_history(history, 'Without Batch Normalization')
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Experiment 4: Change batch size
    for batch_size in [16, 256]:
        print(f"\nExperiment: Batch Size = {batch_size}")
        model = build_cnn()
        history, _, test_accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size=batch_size)
        plot_history(history, f'Batch Size = {batch_size}')
        print(f"Test Accuracy: {test_accuracy:.4f}\n")

if __name__ == "__main__":
    experiment()
