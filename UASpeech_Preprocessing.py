import pandas as pd
from TORGO_Preprocessing import *

"""This file is responsible for preprocessing the UASPEECH dataset and extracting audio files into a data format that
can be fed into deep learning models for training"""


def get_audio_path_UA(speaker):

    return glob("./UASPEECH/audio/**/{}/*.wav".format(speaker), recursive=True)


def get_word_list_UA():

    word_list_xls = pd.read_excel('./UASPEECH/speaker_wordlist.xlsx', sheet_name="Word_filename", header=0)
    word_dictionary = {}

    for i in range(word_list_xls.shape[0]):
        value = word_list_xls.iloc[i].values
        word_dictionary[value[1]] = value[0]

    return word_dictionary


def get_data_UA(wavs):
    data_B1 = []
    data_B2 = []
    data_B3 = []

    word_dictionary = get_word_list_UA()

    for wav in wavs:
        speaker, block, word_key, mic = wav.split('_')
        if word_key.startswith('U'):
            # word_key = '_'.join([block, word_key])
            continue

        text = word_dictionary.get(word_key, -1)
        if text == -1:
            continue
        if block == 'B1':
            data_B1.append({'audio': wav, 'text': text})
        elif block == 'B2':
            data_B2.append({'audio': wav, 'text': text})
        elif block == 'B3':
            data_B3.append({'audio': wav, 'text': text})

    return data_B1, data_B2, data_B3


def get_data_set_UA(speakers, feature_extractor, vectorizer):
    """Extracts and split the data into B1, B2 and B3 as dataset objects that can be used for model training
    :param speakers: list of speakers to get data from UASpeech data_set.
    :param feature_extractor: function which extracts features from data
    :param vectorizer: text vectorizer
    :return: dataset objects for model fitting
    """
    wavs = []
    for speaker in speakers:
        wavs += get_audio_path_UA(speaker)

    data_B1, data_B2, data_B3 = get_data_UA(wavs)

    B1 = create_tf_dataset(data_B1, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=64)      # B1 data
    B2 = create_tf_dataset(data_B2, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=64)     # B2 data
    B3 = create_tf_dataset(data_B3, feature_extractor=feature_extractor, vectorizer=vectorizer, bs=64)  # B3 data
    return B1, B2, B3
