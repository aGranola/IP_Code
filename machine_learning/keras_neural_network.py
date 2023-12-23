import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
import os
import matplotlib.pyplot as plt


import openvsp as vsp
import tensorflow as tf

from ann_visualizer.visualize import ann_viz

cwd = os.getcwd()
print(cwd)

sampleSize = 100

#input data: L/D
#print(cwd + "/openvsp_api/sample_set/wingTest1/wingTest1_DegenGeom.polar")
inputData = []
for i in range(sampleSize):
    currentInputDataset = np.loadtxt(os.path.join(cwd, f"openvsp_api/sample_set/wing_geom_and_analysis_{i}/wing_geom_DegenGeom.polar"), skiprows=1)
    currentInputSampleSet = []
    #read in colomn header to find value (look at read_csv using DictReader)
    currentInputSampleSet.append(currentInputDataset[11])
    inputData.append(currentInputSampleSet)
print(inputData)

#output data
outputData = []
for i in range(sampleSize):
    outputDatasetFilepath = os.path.join(cwd, f"openvsp_api/sample_set/wing_geom_and_analysis_{i}/wing_geom.vsp3")
    vsp.VSPRenew()
    vsp.ReadVSPFile(str(outputDatasetFilepath))
    geoms = vsp.FindGeoms()
    wing = geoms[0]
    #add more parameters
    aspectRatioID = vsp.GetParm(wing, "Aspect", "XSec_1")
    aspectRatio = vsp.GetParmVal(aspectRatioID)
    #print(aspectRatio)
    aspectRatioID = vsp.GetParm(wing, "Aspect", "XSec_1")
    aspectRatio = vsp.GetParmVal(aspectRatioID)
    spanID = vsp.GetParm(wing, "Span", "XSec_1")
    span = vsp.GetParmVal(spanID)
    taperID = vsp.GetParm(wing, "Taper", "XSec_1")
    taper = vsp.GetParmVal(taperID)
    currentOutputSampleSet = []
    currentOutputSampleSet.extend([aspectRatio, span, taper])
    outputData.append(currentOutputSampleSet)
print(outputData)
inputData = np.array(inputData)
outputData = np.array(outputData)

def neural_network_training(
        sampleSize:int = None, 
        inputData:list[float] = None, 
        outputData:list[float] = None, 
        #valInput:list[float] = None, 
        #valOutput:list[float] = None, 
        epochNo:int = 1000, 
        plotLoss:bool = True, 
        visualiseModel:bool = True
    ):

    #split data into 3 groups so the model doesn't overfit
    trainSize = int(sampleSize*.6) #60% of the data is used for training
    valSize = int(sampleSize*.2) #20% is used to validate/improve the model as it is being trained
    testSize = int(sampleSize*.2) + 1 #20% is used to test the model with data nevr seen before

    trainInput, valInput, testInput = np.split(inputData, [trainSize, trainSize + valSize])
    trainOutput, valOutput, testOutput = np.split(outputData, [trainSize, trainSize + valSize])


    model = keras.Sequential()
    model.add(Dense(5, input_shape=(1,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(trainInput, trainOutput, validation_data=(valInput, valOutput), epochs=epochNo, callbacks=[tensorboard_callback])

    if plotLoss == True:
        fig = plt.figure()
        plt.plot(hist.history['loss'], color='blue', label='loss')
        plt.plot(hist.history['val_loss'], color='pink', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig("Loss Curve.png")
        plt.show()

    predictedOutput = model.predict(testInput)
    print(predictedOutput)
    print(testOutput)
    
    if visualiseModel == True:
        ann_viz(model, filename = "Model Shape")

neural_network_training(
    sampleSize = sampleSize, 
    inputData = inputData, 
    outputData = outputData, 
    visualiseModel = False
    #valInput, 
    #valOutput
    )