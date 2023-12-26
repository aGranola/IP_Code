import numpy as np
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from ann_visualizer.visualize import ann_viz

def split_data_for_model(
        inputData:list[float],
        outputData:list[list[float]]
    ):
    sampleSize = len(inputData)
    #split data into 3 groups so the model doesn't overfit
    trainSize = int(sampleSize*.6) #60% of the data is used for training
    valSize = int(sampleSize*.2) #20% is used to validate/improve the model as it is being trained
    testSize = int(sampleSize*.2) + 1 #20% is used to test the model with data never seen before

    trainInput, valInput, testInput = np.split(inputData, [trainSize, trainSize + valSize])
    trainOutput, valOutput, testOutput = np.split(outputData, [trainSize, trainSize + valSize])
    
    return trainInput, valInput, testInput, trainOutput, valOutput, testOutput

# Define the custom activation function
def custom_activation_for_outputs(x):
    normalized_outputs = []
    for i in range(len(x)):
        min_val, max_val = output_ranges[i]
        normalized_output = (x[i] - min_val) / (max_val - min_val)
        normalized_outputs.append(normalized_output)
    return tf.stack(normalized_outputs)  # Ensure TensorFlow tensor format


def create_neural_network(
        trainOutput:list[list[float]],
        output_ranges
    ):
    #  get num variables in output from output data
    model_output_dim = len(trainOutput[0])
    model = keras.Sequential()
    model.add(Dense(5, input_shape=(1,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(model_output_dim, activation=lambda x: custom_activation_for_outputs(x, output_ranges)))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Custom callback 
class OutputMonitor(Callback):
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs=None):
        train_predictions = self.model.predict(self.training_data)
        test_predictions = self.model.predict(self.test_data)

        # Analyze output values, print or log them
        print(f"Epoch {epoch + 1}:")
        print("Training outputs:", train_predictions)
        print("Test outputs:", test_predictions)

def train_neural_network(model, trainInput, trainOutput, valInput, valOutput, testInput, epochNum):
    logdir='logs'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    monitor = OutputMonitor(trainInput, testInput)
    
    hist = model.fit(
        trainInput,
        trainOutput,
        validation_data=(valInput, valOutput),
        epochs=epochNum,
        callbacks=[tensorboard_callback, monitor]
    )    
    return hist

def plot_loss(hist):
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='blue', label='loss')
    plt.plot(hist.history['val_loss'], color='pink', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("Loss Curve.png")
    plt.show()