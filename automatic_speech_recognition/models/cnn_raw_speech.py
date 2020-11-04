import tensorflow as tf
from tensorflow.keras import layers


def get_layers(context = 300, filt_length_conv_1 = 150, filt_length_conv_2 = 5, 
              number_filters_conv_1 = 100, number_filters_conv_2 = 80, fc_cells = 512):
    '''
    Returns the paramters that for a CNN based model.
    Consider chaning default parameters as per application.
    
    Architecture: CNN->FC->Logits
    
    Reference:
    "Analysis of CNN-based Speech Recognition System using Raw Speech as Input"
    (https://ronan.collobert.com/pub/matos/2015_cnnspeech_interspeech.pdf)
    '''

    y_true = tf.keras.layers.Input((None,), name="y_true")
    y_true_length = tf.keras.layers.Input((1), name="y_true_length")

    # Inputshape [btch_sz, n_time, n_channels = 1]
    input_audio = layers.Input(shape=(None, 1), name="audio_input")

    # Append zeros in time for context at the beggining and end of audio sample
    input_audio_padded = layers.ZeroPadding1D(padding=(context), name = 'zero_padding_context')(input_audio)

    # Apply 1d filter: Shape [btch_sz, n_time, n_channels]
    filt_length_conv_1 = 2 * context + 1 + filt_length_conv_1
    conv = layers.Conv1D(number_filters_conv_1, filt_length_conv_1, strides=100, activation='relu', name="conv_1")(input_audio_padded)
    conv = layers.Conv1D(number_filters_conv_2, filt_length_conv_2, strides = 1, activation='relu', name="conv_2")(conv)
    conv = layers.Conv1D(number_filters_conv_2, filt_length_conv_2, strides = 1, activation='relu', name="conv_3")(conv)

    # Apply FC layer to each time step
    fc = layers.TimeDistributed(layers.Dense(fc_cells), name = 'fc')(conv)
    fc = layers.ReLU()(fc)
    fc = layers.Dropout(rate=0.1)(fc)

    # Apply FC layer to each time step to output prob distro across chars
    logits = layers.TimeDistributed(layers.Dense(29, activation='softmax'), name = 'logits')(fc)

    return logits, input_audio, y_true, y_true_length

