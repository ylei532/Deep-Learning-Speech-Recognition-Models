from glob import glob
import librosa
import tensorflow as tf
import numpy as np
import re
import math
import random

"""This file is responsible for preprocessing the TORGO dataset and extracting audio files into a data format that
can be fed into deep learning models for training"""


# Global variables that control parameters for feature extraction
SAMPLING_RATE = 22050
HOP_LENGTH = 100
FRAME_LENGTH = 512
MEL = False     # Set true to extract mel spectrogram (instead of normal spectrogram)
N_MFCC = 13
ORDER = 1
PAD_LEN = 10
MFCC = False    # must set MEL to True also to get MFCC's


def get_audio_path(speaker):
    """
    Returns path to audio files belonging to specified speaker
    :param speaker: string
                       string encoding the speaker
    :return: list of strings
                the strings represent file paths.
    """
    return glob("./TORGO/datasets/{}/**/wav_*/*.wav".format(speaker))


def get_data(wavs, maxlen=5000):
    """
    Returns a mapping of audio paths to text
    ---
    :param wavs: string
                string containing path to audio file,
    :param maxlen: int
                max length of word
    :return data: list of dictionaries
                each dictionary contain "audio" and "text", corresponding to the audio path and its text
            removed_files: list of files that were excluded from data
    """
    data = []
    removed_files = []

    pattern = re.compile(r"\[.*\]")
    for wav in wavs:
        description = wav.split("/")
        session = description[4]
        id = description[-1].split('.')[0]
        speaker = description[3]
        try:
            filename = glob(f"./TORGO/datasets/{speaker}/{session}/prompts/{id}.txt")[0]
        except IndexError:
            continue
        with open(filename, encoding="utf-8") as f:
            line = f.readline()
            line = line.replace("\n", "")
            if len(line) > maxlen:
                continue
            line = pattern.sub("", line)
            if line == "" or line == "xxx" or '.jpg' in line:
                removed_files.append({'file': filename, 'text': line})
                continue
            line = line.rstrip()
            data.append({"audio": wav, "text": line})
            random.shuffle(data)
    return data, removed_files


class VectorizeChar:
    """Class used to vectorize dataset into a sequence of integers and create token index"""

    def __init__(self, max_len=50):
        self.vocab = (
                ["-", "<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + [" ", ".", ",", "?", "'"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[:self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def audio_file_to_feature(path, pad_len=PAD_LEN, sr=SAMPLING_RATE, hop_length=HOP_LENGTH, frame_size=FRAME_LENGTH, mel=MEL, mfcc=MFCC):
    """
    Extracts features from audio files using the TensorFlow library
    ---
    :param path: string
                    path to audio file
    :param pad_len: int
                       pad output sequences to 'pad' seconds long
    :param sr: int
                  Sampling rate used to sample data from audio file
    :param hop_length: int
                          Number of overlapping samples between frames
    :param frame_size: Int
                          Size of frame to apply the fourier transform on data
    :param mel: Boolean
                   Set true to extract mel spectrogram
    :param mfcc: Boolean
                    Set true to extract MFCC
    :return: specified audio feature from audio sample (as a TF object)
    """
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)    # return floating poiht time series of shape (samples, 1)
    audio = tf.squeeze(audio, axis=-1)  # reshape audio data to (samples,)
    stfts = tf.signal.stft(audio, frame_length=frame_size, frame_step=hop_length, fft_length=frame_size)  # returns (frames, freq_bin). frames = (samples - frame_length) / hop_length. freq_bin = fft_length / 2 +1

    if mel:
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, SAMPLING_RATE, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        x = tf.math.log(mel_spectrograms + 1e-6)
        if mfcc:
            x = tf.signal.mfccs_from_log_mel_spectrograms(x)[:, :N_MFCC]
    else:
        x = tf.math.pow(tf.abs(stfts), 0.5)

    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs

    # padding to 10 seconds
    pad_len = math.ceil((pad_len / (1 / sr) - frame_size) / hop_length)
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x    # returns (frames, freq_bin), where values have been normalized, and equivalent to a 10s audio clip


def audio_file_to_spectrogram(path, sr=SAMPLING_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, win_length=FRAME_LENGTH, pad_len=PAD_LEN, mel=MEL):
    """
    Extracts spectrogram from audio file
    ---
    :param path: string
                    path to audio file
    :param pad_len: int
                       pad output sequences to 'pad' seconds long
    :param sr: int
                  Sampling rate used to sample data from audio file
    :param hop_length: int
                          Number of overlapping samples between frames
    :param n_fft: Int
                     Number of samples to apply the fourier transform on data at once
    :param win_length: Int
                     Number of samples to bundle in a frame
    :param mel: Boolean
                   Set true to extract mel spectrogram
    :return: (Mel)Spectrogram for audio sample
    """

    audio, _ = librosa.load(path, sr=sr)
    if mel:
        audio_spectrogram = librosa.feature.melspectrogram(audio, n_fft=n_fft, hop_length=hop_length,
                                                           win_length=win_length)
    else:
        audio_spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    audio_spectrogram = np.abs(audio_spectrogram.T) ** 0.5
    audio_spectrogram = tf.Variable(initial_value=audio_spectrogram)

    # normalizing data
    means = tf.math.reduce_mean(audio_spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(audio_spectrogram, 1, keepdims=True)
    audio_spectrogram = (audio_spectrogram - means) / stddevs

    # padding timeseries to standardized length
    pad_len = math.ceil((pad_len / (1 / sr) - n_fft) / hop_length)
    paddings = tf.constant([[0, pad_len], [0, 0]])
    audio_spectrogram = tf.pad(audio_spectrogram, paddings, "CONSTANT")[:pad_len, :]

    return audio_spectrogram.numpy()


def audio_file_to_mfcc(path, n_mfcc=N_MFCC, sr=SAMPLING_RATE, hop_length=HOP_LENGTH, pad_len=PAD_LEN, order=ORDER):
    """
    Extract mfcc from audio file
    ---
    :param path: string
                    path to audio file
    :param pad_len: int
                       pad output sequences to 'pad' seconds long
    :param sr: int
                  Sampling rate used to sample data from audio file
    :param hop_length: int
                          Number of overlapping samples between frames
    :param n_mfcc: Int
                     # of coefficients to extract in the mfcc
    :param order: Int (either 0, 1 or 2)
                     number of derivatives to extract values from
    :return: MFCC for audio sample
    """

    # path = path.numpy().decode('utf-8')
    audio, _ = librosa.load(path, sr=sr)
    mfccs = (librosa.feature.mfcc(audio, n_mfcc=n_mfcc, sr=sr, hop_length=HOP_LENGTH))
    if order == 2:
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    elif order == 1:
        delta_mfccs = librosa.feature.delta(mfccs)
        mfccs = np.concatenate((mfccs, delta_mfccs))
    mfccs = mfccs.T
    mfccs = tf.Variable(initial_value=mfccs)

    # normalizing data
    means = tf.math.reduce_mean(mfccs, 1, keepdims=True)
    stddevs = tf.math.reduce_std(mfccs, 1, keepdims=True)
    mfccs = (mfccs - means) / stddevs

    # padding timeseries to standardized length
    pad_len = math.floor((pad_len / (1 / sr)) / hop_length)
    paddings = tf.constant([[0, pad_len], [0, 0]])
    mfccs = tf.pad(mfccs, paddings, "CONSTANT")[:pad_len, :]

    return mfccs.numpy()


def audio_file_to_time_domain_features(path, frame_size=FRAME_LENGTH, hop_length=HOP_LENGTH, sr=SAMPLING_RATE, pad_len=PAD_LEN):
    """ Extract time domain features from audio file in the format (amplitude envelope, RMSE, zero crossing rate)"""
    # Amplitude envelope extraction
    amplitude_envelope = []
    audio, _ = librosa.load(path, sr=sr)

    # Calculate AE for each frame
    for i in range(0, len(audio), hop_length):
        current_frame_ae = max(audio[i:i+frame_size])
        amplitude_envelope.append(current_frame_ae)
    amplitude_envelope = np.array(amplitude_envelope).reshape(len(amplitude_envelope), 1)

    # Calculate RMSE for each frame
    rms = librosa.feature.rms(audio, frame_length=frame_size, hop_length=hop_length)[0]
    dlen = len(rms) - amplitude_envelope.shape[0]
    if dlen != 0:
        rms = np.delete(rms, np.s_[-1*dlen:], axis=-1)
    rms = rms.reshape(len(amplitude_envelope), 1)

    # Calculate zero crossing rate for each frame
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_size, hop_length=hop_length)[0]
    dlen = len(zcr) - amplitude_envelope.shape[0]
    if dlen != 0:
        zcr = np.delete(zcr, np.s_[-1*dlen], axis=-1)
    zcr = zcr.reshape(len(amplitude_envelope), 1)
    time_domain_features = np.concatenate((amplitude_envelope, rms, zcr), axis=-1)
    time_domain_features = tf.Variable(initial_value=time_domain_features)

    # padding timeseries to standardized length
    pad_len = math.floor((pad_len / (1 / sr)) / hop_length)
    paddings = tf.constant([[0, pad_len], [0, 0]])
    time_domain_features = tf.pad(time_domain_features, paddings, "CONSTANT")[:pad_len, :]

    return time_domain_features.numpy()


def create_tf_dataset(data, feature_extractor, vectorizer, bs=128):
    """
    Create tensorflow dataset to feed into transformer model
    ---
    :param data: dictionary with keys 'audio' and 'text
                'audio' contains string of audio file path, and 'text' is a string with the corresponding text
    :param bs: int
              number of samples per batch
    :param feature_extractor: function
                                determines the features extracted from the audio data
    :param vectorizer: VectorizerChar object
    :return: tensorflow dataset object
    """
    # create text_ds
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)

    # create audio_ds
    flist = [_["audio"] for _ in data]
    if feature_extractor == audio_file_to_feature:
        audio_ds = tf.data.Dataset.from_tensor_slices(flist)
        audio_ds = audio_ds.map(audio_file_to_feature)
    else:
        audio_ds = list(map(feature_extractor, flist))
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_ds)

    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_data_set_TORGO(speakers, feature_extractor, vectorizer, train_test_val_split=0.8):
    """
    Gets training and validation data set
    :param speakers: list of speakers to get data from TORGO data_set.
    :param train_test_val_split: split ratio for train and test set
    :param feature_extractor: function which extracts features from data
    :param vectorizer: text vectorizer
    :return: training data set and validation dataset as dataset objects
    """
    wavs = []
    for speaker in speakers:
        wavs += get_audio_path(speaker)

    data, _ = get_data(wavs)
    split = int(len(data) * train_test_val_split)
    train_data = data[:split]
    val_data = data[split:split+int((len(data)-split)/2)]
    test_data = data[split+int((len(data)-split)/2):]

    ds = create_tf_dataset(train_data, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=64)
    val_ds = create_tf_dataset(val_data, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=6)
    test_ds = create_tf_dataset(test_data, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=6)
    return ds, val_ds, test_ds

