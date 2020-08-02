import wave
import struct
import os
import bz2
import pickle
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from collections import OrderedDict
from scipy import fft
from scipy import signal
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm  # tqdm provides progress bar utility
import pydot as pyd
import librosa


#Initialise list with labels.
command_words = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine'
]

## Default Config for running on GPU
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

#Short-time Fourier transform Function
#Input: raw data, output: Frequency, Time and Stft 
#Padding is added if the input is not 1 second in length
#Hann Function is applied to the signal
def stft_windowed(audio, sampling_rate, duration, window=20, stride=10):
    n = len(audio)
    T = 1/sampling_rate
    d = 4  #arr.strides[0]

    stride_size = int(0.001 * sampling_rate * stride)
    window_size = int(0.001 * sampling_rate * window)
    total_samples = int(sampling_rate * duration)

    if n > total_samples:
        audio = audio[0:total_samples]
    else:
        padding = total_samples - n   # add padding at both ends
        offset = padding // 2
        audio = np.pad(audio, (offset, total_samples - n - offset), 'constant')
        n = len(audio)

    f, t, stft = signal.stft(audio, sampling_rate, window='hann',
                             nperseg=window_size - 1, noverlap=stride_size)
    return (f, t, stft)


#Read in audio files from file destination
def read_wav_file(path):
    wave_file = wave.open(path)
    wave_data = wave_file.readframes(wave_file.getnframes())
    # unpack short data stored in byte. Format string is "little eindian, num of samples, short type"
    data = struct.unpack("<" + str(wave_file.getnframes()) + "h", wave_data)
    return data

#Function for preprocessing data
#Calls "stft_windowed" function
#Uses Mel-Frequency scaling
#Data is transposed to get frequency by time, outputted as dictionary
def preprocess(dict, parent_dir, path):
    if parent_dir not in dict:
        dict[parent_dir] = []
    raw_data = read_wav_file(str(path))
    f, t, processed_data = stft_windowed(raw_data, 16000, 1)
    mel = librosa.feature.melspectrogram(S=processed_data, sr=16000)
    f = f
    t = t[2:]
    processed_data = processed_data.T[2:]
    mel = mel.T[2:]
    mel = np.abs(mel)
    dict[parent_dir].append(mel)

#Function to one-hot encode labels
#Output is a 2-d Array 
def encode(arr):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(np.array(arr))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

#This section of code checks to see if the dataset has been prepared or not previously, saves time running tests
pathlist = Path("dataset").glob('**/*.wav')

#If the files don't exist, create them.
if not os.path.isfile("train.npz") or not os.path.isfile("test.npz") or not os.path.isfile("validation.npz"):
    training_dict = {}
    testing_dict = {}
    validation_dict = {}
    data_dict = {}

    with open("dataset/testing_list.txt", "r") as testing_list:
        testing_manifest = [line.rstrip() for line in testing_list]
    with open("dataset/validation_list.txt", "r") as validation_list:
        validation_manifest = [line.rstrip() for line in validation_list]

    print("Converting input audio to samples. This may take a while.")

    for path in tqdm(pathlist):
        path_str = str(path).replace("\\", "/").replace("dataset", "")[1:]
        parent_dir = path_str.split("/")[0]
        if parent_dir in command_words:
            if path_str in testing_manifest:
                preprocess(testing_dict, parent_dir, path)
            elif path_str in validation_manifest:
                preprocess(validation_dict, parent_dir, path)
            else:
                preprocess(training_dict, parent_dir, path)

    print("Saving processed data to file.")

    print("Writing validation dataset...")
    np.savez("validation", validation_dict)
    print("Writing test dataset...")
    np.savez("test", testing_dict)
    print("Writing train dataset...")
    np.savez("train", training_dict)
else:
    print("Loading existing data...")
    validation_dict = np.load("validation.npz", allow_pickle=True)
    testing_dict = np.load("test.npz", allow_pickle=True)
    training_dict = np.load("train.npz", allow_pickle=True)
    training_dict = training_dict.f.arr_0[()]
    testing_dict = testing_dict.f.arr_0[()]
    validation_dict = validation_dict.f.arr_0[()]

#Initislise algorithm inputs
X_train = []
X_test = []
Y_train = []
Y_test = []

#Populate values from each sample to their associated dictionary
for key, value in training_dict.items():
    for array in value:
        X_train.append(array)
    Y_train.extend([key for i in range(len(value))])
for key, value in testing_dict.items():
    for array in value:
        X_test.append(array)
    Y_test.extend([key for i in range(len(value))])

#Convert dictionary objects to arrays.
X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = encode(Y_train)
Y_test = encode(Y_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

#The attention function used by the transformer takes three inputs: Q (query), K (key), V (value)
#Calucated the attention weights, must have matchings shape dimensions
#Outputs the attention weights
def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)

#Each multi-head attention block gets three inputs; Q (query), K (key), V (value). 
#These are put through linear (Dense) layers and split up into multiple heads.
#The attention output for each head is then concatenated
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(
            query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs

#Class Provides the model with information about the relative position of the input values
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) /
                            tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#This allows to the transformer to know where there is real data and where it is padded
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


#Encoder layer consists of Multi-head attention with padding & feedforward networks
#Takes in values defined in transfromer object:
   #time_steps=time_steps,
   #num_layers=num_layers,
   #units=units,
   #d_model=d_model,
   #num_heads=num_heads,
   #dropout=dropout,
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

#Encoder Layer:
#linear projection, positional encoding and the 4 encoder layers.
#The output from the encoder layer is passed into a series of convolutional layers alongside the one hot encoded labels
def encoder(time_steps,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    projection = tf.keras.layers.Dense(
        d_model, use_bias=True, activation='linear')(inputs)
    projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    projection = PositionalEncoding(time_steps, d_model)(projection)

    outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

#Defining the model architecture.
def transformer(time_steps,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                output_size,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(tf.dtypes.cast(

            # Like our input has a dimension of length X d_model but the masking is applied to a vector
            # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked
            tf.math.reduce_sum(
                inputs,
                axis=2,
                keepdims=False,
                name=None), tf.int32))

    enc_outputs = encoder(
        time_steps=time_steps,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='encoder'
    )(inputs=[inputs, enc_padding_mask])
  
    static = tf.reshape(enc_outputs,(-1, time_steps, d_model))
    resh = tf.keras.backend.expand_dims(static, -1)
    conv = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(resh)
    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(pool)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = tf.keras.layers.Flatten()(pool2)

    dense = tf.keras.layers.Dense(128, activation='relu')(flat)
    
    # Prediction
    outputs = tf.keras.layers.Dense(units=output_size, use_bias=True, activation='softmax', name="outputs")(dense)

    return tf.keras.Model(inputs=[inputs], outputs=outputs, name='voice_classification')

## Making sure there are no INF/Nan values in the data which leads to issues during training/classification
assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(Y_train))
assert not np.any(np.isnan(X_test))
assert not np.any(np.isnan(Y_test))
assert not np.any(np.isinf(X_train))
assert not np.any(np.isinf(Y_train))
assert not np.any(np.isinf(X_test))
assert not np.any(np.isinf(Y_test))

#HyperParameters for model
NUM_LAYERS = 4
D_MODEL = 128
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.1
TIME_STEPS = 100
OUTPUT_SIZE = 10
EPOCHS = 100

model = transformer(
    time_steps=TIME_STEPS,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    output_size=OUTPUT_SIZE)

#The Adam optimiser
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

#Plotting model
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)

print(model.summary())

history = model.fit(X_train, Y_train, epochs=EPOCHS,
                    validation_data=(X_test, Y_test))

history_dict = history.history

#Plotting accuracy and loss of model.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()