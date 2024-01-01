import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.callbacks import Callback
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
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
def custom_activation_for_outputs(x, min_values, max_values):
    """Normalizes each output node between its corresponding min and max values."""
    min_values = K.constant(min_values)  # Convert lists to tensors
    max_values = K.constant(max_values)

    normalized = (x - K.min(x, axis=-1, keepdims=True)) / (K.max(x, axis=-1, keepdims=True) - K.min(x, axis=-1, keepdims=True))
    return normalized * (max_values - min_values) + min_values

def create_neural_network(
        numOutputValues:int,
        output_ranges
    ):
    min_values = [ min_max[0] for min_max in output_ranges]
    max_values = [ min_max[1] for min_max in output_ranges]
    
    model = Sequential([
        Dense(5, input_shape=(1,), activation='relu'),
        Dense(5, activation='relu'),
        Dense(numOutputValues),  # Output layer without activation
        Lambda(
            custom_activation_for_outputs,
            arguments={
                'min_values': min_values,
                'max_values': max_values
            }
        )
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Custom callback 
class OutputMonitor(Callback):
    def __init__(self, training_input, training_output):
        self.training_input = training_input
        self.training_output = training_output
    
    def on_epoch_end(self, epoch, logs=None):
        train_predictions = self.model.predict(self.training_input)

        # Print output of first 3 outputs
        print(f"Epoch {epoch + 1}:")
        print("Training inputs:",  np.vectorize("{:.2f}".format)(self.training_input)[:3])
        print("Training outputs:",  np.vectorize("{:.2f}".format)(self.training_output)[:3])
        print("Predicted outputs", np.vectorize("{:.2f}".format)(train_predictions)[:3])

def train_model(model, trainInput, trainOutput, valInput, valOutput, numEpochs, batchSize):
    logdir='logs'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    monitor = OutputMonitor(trainInput, trainOutput)

    hist = model.fit(
        trainInput,
        trainOutput,
        validation_data=(valInput, valOutput),
        epochs=numEpochs,
        batchSize=batchSize,
        callbacks=[tensorboard_callback, monitor]
    )    
    return hist


###########################  VAE #######################################
def vae_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def mse_vae_loss(encoder_input, decoder_output, z_mean, z_log_var, output_dim):
    reconstruction_loss = mean_squared_error(encoder_input, decoder_output) * output_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss

def kl_divergence_vae_loss(encoder_input, decoder_output, z_mean, z_log_var, beta=1.0):
    reconstruction_loss = K.mean(K.square(encoder_input - decoder_output), axis=-1)
    kl_divergence = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1), axis=-1)
    return reconstruction_loss + (beta * kl_divergence)  # `beta` controls KL weight


def create_vae(
        outputDim:int,
        outputRanges: list[tuple[int]]
    ):
    min_values = [ min_max[0] for min_max in outputRanges]
    max_values = [ min_max[1] for min_max in outputRanges]
    
    # Encoder
    input_dim = 1  # Adjust as per your input
    latent_dim = 2  # Latent space dimension
    encoder_input = Input(shape=(input_dim,))  # 1 input
    # Encoder network
    encoded = Dense(64, activation='relu')(encoder_input)
    # encoded = Dense(100, activation='relu')(encoded)
    z_mean = Dense(latent_dim, name='z_mean')(encoded) # Latent mean
    z_log_var = Dense(latent_dim, name='z_log_var')(encoded) # Latent log variance
    # Latent sampling layer
    z = Lambda(vae_sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    # Decoder
    decoder_input = Input(shape=(latent_dim,))
    # Decoder network
    decoded = Dense(64, activation='relu')(decoder_input)
    # decoded = Dense(50, activation='relu')(decoded)
    decoded = Dense(outputDim, activation='sigmoid')(decoded)
    decoder_output = Lambda(
        custom_activation_for_outputs,
        arguments={
            'min_values': min_values,
            'max_values': max_values
        }
    )(decoded)
    decoder = Model(decoder_input, decoder_output, name='decoder')
    
    # VAE
    output = decoder(encoder(encoder_input)[2])
    vae = Model(encoder_input, output, name='vae')
    # vae_loss = kl_divergence_vae_loss(encoder_input, output, z_mean, z_log_var, beta=1.0)
    # vae_loss = mse_vae_loss(encoder_input, output, z_mean, z_log_var, outputDim)
    # vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return vae

def create_gan(
        outputDim:int,
        outputRanges: list[tuple[int]]
    ):
    min_values = [ min_max[0] for min_max in outputRanges]
    max_values = [ min_max[1] for min_max in outputRanges]

    # Generator
    # Input layer for single input value
    generator_input = Input(shape=(1,))

    # Hidden layers for feature learning (adjust as needed)
    x = Dense(128, activation='relu')(generator_input)
    x = Dense(256, activation='relu')(x)

    # Output layer with 7 outputs
    generator_output = Dense(outputDim, activation='sigmoid')(x)  # Sigmoid for constraining outputs

    # Create the generator model
    generator = Model(generator_input, generator_output)
    
    # Discriminator
    discriminator_input = Input(shape=(7,))

    # Hidden layers for discrimination (adjust as needed)
    x = Dense(256, activation='relu')(discriminator_input)
    x = Dense(128, activation='relu')(x)

    # Output layer with sigmoid activation for binary classification
    discriminator_output = Dense(1, activation='sigmoid')(x)

    # Create the discriminator model
    discriminator = Model(discriminator_input, discriminator_output) 
    # Freeze discriminator's weights when training the generator
    discriminator.trainable = False

    # create gan
    gan_input = Input(shape=(1,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)  # Common choice for GANs

    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')
    return generator, discriminator, gan
    
def train_gan(numEpochs:int, batchSize:int, trainInput, trainOutput, generator, discriminator, gan):
    discriminator_losses = []
    generator_losses = []
    for epoch in range(numEpochs):
        # Train discriminator on real and fake data
        for batch_i in range(0, len(trainInput), batchSize):
            real_inputs = trainInput[batch_i:batch_i+batchSize]
            real_outputs = trainOutput[batch_i:batch_i+batchSize]

            generated_outputs = generator.predict(real_inputs)
            # Calculate discriminator loss (average of real and fake losses)
            discriminator_loss_real = discriminator.train_on_batch(real_outputs, np.ones((len(real_outputs), 1))) # Real labels as 1
            discriminator_loss_fake = discriminator.train_on_batch(generated_outputs, np.zeros((len(generated_outputs), 1))) # Fake labels as 0
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            discriminator_losses.append(discriminator_loss)


        # Train generator (using frozen discriminator)
        noise = np.random.normal(0, 1, (batchSize, 1))  # Add noise for better exploration
        discriminator.trainable = False
        generator_loss = gan.train_on_batch(noise, np.ones((batchSize, 1)))  # Flip labels for generator
        generator_losses.append(generator_loss)
        discriminator.trainable = True
        print(f"Epoch {epoch+1}, Discriminator Loss: {discriminator_loss:.4f}, Generator Loss: {generator_loss:.4f}")
    return discriminator_losses, generator_losses

