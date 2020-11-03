# Read audio file
def _decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

# Get the label of the audio file
def _get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep, result_type = "RaggedTensor")[-2]
    return parts

def _where(char):
    toks = [" ","a","b","c","d","e",
            "f","g","h","i","j","k",
            "l","m","n","o","p","q",
            "r","s","t","u","v","w",
            "x","y","z","'","-"]
    return tf._where(tf.equal(char, toks))

# For a set of chars, get the idxs from the toks list
def _tokenize_label(split_label):
    split_label = tf.convert_to_tensor(split_label)
    split_into_toks = tf.map_fn(_where, split_label, dtype=np.int64)
    return split_into_toks[:,0,0]

# Read audio from Path
def _get_waveform_and_label(file_path):
    label = _get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = _decode_audio(audio_binary)
    return waveform, label, tf.strings.length(label)

def preprocess_simple_word_dataset(files):
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