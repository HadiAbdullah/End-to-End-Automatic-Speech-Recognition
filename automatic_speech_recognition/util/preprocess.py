import os
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav


def _decode_audio(audio_binary):
    '''
    Read audio file
    '''
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def _get_label(file_path):
    '''
    Get the label of the audio file
    '''
    parts = tf.strings.split(file_path, os.path.sep, result_type = "RaggedTensor")[-2]
    return parts

def _where(char):
    toks = [" ","a","b","c","d","e",
            "f","g","h","i","j","k",
            "l","m","n","o","p","q",
            "r","s","t","u","v","w",
            "x","y","z","'","-"]
    return tf.where(tf.equal(char, toks))

def _tokenize_label(split_label):
    '''
    For a set of chars, get the idxs from the toks list
    '''
    split_label = tf.convert_to_tensor(split_label)
    split_into_toks = tf.map_fn(_where, split_label, dtype = tf.int64)
    return split_into_toks[:,0,0]

def _get_waveform_and_label(file_path):
    '''
    Read audio from Path
    '''
    
    label = _get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = _decode_audio(audio_binary)
    return waveform, label, tf.strings.length(label)

def preprocess_simple_word(files):
    '''
    Read audio from a file paths.
    Return a dataset object for that file.

    files: A tensor array of paths to the audio file. 
    '''
    files_ds = tf.data.Dataset.from_tensor_slices(files)

    # Read the audio from files
    waveform_ds = files_ds.map(_get_waveform_and_label, num_parallel_calls=-1)
    
    # For each label, split it into chars from string
    waveform_ds = waveform_ds.map(lambda audio, label, label_len: 
                                  (audio, tf.strings.bytes_split(label), label_len))
    # convert tokenize each label
    waveform_ds = waveform_ds.map(lambda audio, label_chars, label_len: 
                                  (audio, _tokenize_label(label_chars), label_len))

    # Add a channel dim to the audio
    waveform_ds = waveform_ds.map(lambda audio, label_tokenized, label_len: 
                                  (tf.expand_dims(audio, -1), label_tokenized, label_len))

    # Map to dictionary so its works with the model input layers
    waveform_ds = waveform_ds.map(lambda audio, label_tokenized, label_len: 
                                  {'audio_input' : audio, 
                                   'y_true' : label_tokenized, 
                                   'y_true_length' : label_len})

    return waveform_ds


def read_simple_word(folder_path):
    '''
    Reads the audio file paths in the folder_path
    '''
    commands = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
#     commands = np.array(tf.io.gfile.listdir(str(folder_path)))
#     print(commands)
    filenames = []
    for command in commands:
#         print(str(folder_path) + command)
        filenames.append(tf.io.gfile.glob(str(folder_path) + command +'/*')) 
    filenames = [item for sublist in filenames for item in sublist]
    
    filenames = tf.random.shuffle(filenames)
    return filenames


def read_audio(file_path):
    """ 
    Read audio from file_path 
    """
    fs, audio = wav.read(file_path)
    return audio
