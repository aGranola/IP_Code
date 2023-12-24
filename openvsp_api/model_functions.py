import numpy as np
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
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


def train_neural_network(
        trainInput:list[float],
        trainOutput:list[list[float]],
        valInput:list[float],
        valOutput:list[list[float]],
        epochNo:int = 1000,
        visualiseModel:bool = True
    ):

    #  get num variables in output from output data
    model_output_dim = len(trainOutput[0])
    model = keras.Sequential()
    model.add(Dense(5, input_shape=(1,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(model_output_dim, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(
        trainInput,
        trainOutput,
        validation_data=(valInput, valOutput),
        epochs=epochNo,
        callbacks=[tensorboard_callback]
    )
    
    if visualiseModel == True:
        ann_viz(model, filename = "Model Shape")
    
    return hist

def plot_loss(hist):
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='blue', label='loss')
    plt.plot(hist.history['val_loss'], color='pink', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("Loss Curve.png")
    plt.show()