import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers


def get_layers(context = 300, filt_length_conv_1 = 150, filt_length_conv_2 = 5, 
              number_filters_conv_1 = 40, number_filters_conv_2 = 256, LSTM_cells = 832, fc_cells = 512):
    '''
    Returns the paramters that for a CNN based model.
    Consider changing default parameters as per application.
    
    Architecture: CNN->CNN->LSTM->LSTM->LSTM->FC->Logits
    
    Reference:
    "Learning the Speech Front-end With Raw Waveform CLDNNs"
    (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43960.pdf)
    '''

    y_true = tf.keras.layers.Input((None,), name="y_true")
    y_true_length = tf.keras.layers.Input((1), name="y_true_length")

    # Inputshape [btch_sz, n_time, n_channels = 1]
    input_audio = layers.Input(shape=(None, 1), name="audio_input")

    # Append zeros in time for context at the beggining and end of audio sample
    input_audio_padded = layers.ZeroPadding1D(padding=(context), name = 'zero_padding_context')(input_audio)

    # Apply 1d filter: Shape [btch_sz, n_time, n_channels]
    filt_length_conv_1 = 2 * context + 1 + filt_length_conv_1
    conv = layers.Conv1D(number_filters_conv_1, 
                         filt_length_conv_1, 
                         kernel_initializer = tf.glorot_normal_initializer(),
                         strides=100, 
                         activation='relu', 
                         name="conv_1")(input_audio_padded)
    conv = layers.Conv1D(number_filters_conv_2, 
                         filt_length_conv_2, 
                         kernel_initializer = tf.glorot_normal_initializer(),
                         strides = 1, 
                         activation='relu', 
                         name="conv_2")(conv)

    # 3 Layers of LSTMS
    rnn = tf.keras.layers.LSTM(LSTM_cells, 
                               kernel_initializer = initializers.RandomUniform(minval=-0.02, maxval=0.02),
                               return_sequences = True, 
                               name = 'rnn_1')(conv)
    rnn = tf.keras.layers.LSTM(LSTM_cells, 
                               kernel_initializer = initializers.RandomUniform(minval=-0.02, maxval=0.02),
                               return_sequences = True, 
                               name = 'rnn_2')(rnn)
    rnn = tf.keras.layers.LSTM(LSTM_cells, 
                               kernel_initializer = initializers.RandomUniform(minval=-0.02, maxval=0.02),
                               return_sequences = True, 
                               name = 'rnn_3')(rnn)
    # Apply FC layer to each time step
    fc = layers.TimeDistributed(layers.Dense(fc_cells,
                                             kernel_initializer = tf.glorot_normal_initializer(),
                                            ), name = 'fc')(rnn)
    fc = layers.ReLU()(fc)
    fc = layers.Dropout(rate=0.1)(fc)

    # Apply FC layer to each time step to output prob distro across chars
    logits = layers.TimeDistributed(layers.Dense(29,
                                                 kernel_initializer = tf.glorot_normal_initializer(),
                                                 activation='softmax'), name = 'logits')(fc)


    return logits, input_audio, y_true, y_true_length

