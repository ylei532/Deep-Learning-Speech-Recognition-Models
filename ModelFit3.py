import keras.callbacks
import os
import keras_tuner as kt
from UASpeech_Preprocessing import *
# from TransformerModel import *
from TransformerModel2 import *
# from TransformerModel4 import *


"""
NOTES:
     - This file is used for applying transfer learning to the Transformer models.
     - Uncomment/comment imports to access the required DL model. ONLY have ONE model uncommented at a time. 
     - For more information, refer to notes in ModelFit.py and ModelFit2.py
"""

# global variables used to determine which layers to freeze
encoder_layers_to_freeze = 3
SpeechEmbeddingConv1_trainable = False
SpeechEmbeddingConv2_trainable = False
SpeechEmbeddingConv3_trainable = True

strategy = tf.distribute.MultiWorkerMirroredStrategy()
maxlen = 50
vectorizer = VectorizeChar(maxlen)
CONTROL_SPEAKERS = ['CF02', 'CF03', 'CF04', 'CF05', 'CM01', 'CM04', 'CM05', 'CM06', 'CM08', 'CM09', 'CM10']
SPEAKERS = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04', 'M05', 'M07', 'M08', 'M09', 'M12', 'M16']

ds_global = []        # Store B1 data from UASPEECH speakers
val_ds_global = []    # Store B2 data from UASPEECH speakers
test_ds_global = []   # store B3 data from UASPEECH speakers

control_ds_global = []        # Store B1 data from UASPEECH control speakers
control_val_ds_global = []    # Store B2 data from UASPEECH control speakers
control_test_ds_global = []   # store B3 data from UASPEECH control speakers

for i in range(len(SPEAKERS)):
    speaker_ds, speaker_val_ds, speaker_test_ds = get_data_set_UA([SPEAKERS[i]], feature_extractor=audio_file_to_feature, vectorizer=vectorizer)
    ds_global.append(speaker_ds)            # B1 data
    val_ds_global.append(speaker_val_ds)    # B2 data
    test_ds_global.append(speaker_test_ds)  # B3 data


for i in range(len(CONTROL_SPEAKERS)):
    speaker_ds, speaker_val_ds, speaker_test_ds = get_data_set_UA([CONTROL_SPEAKERS[i]], feature_extractor=audio_file_to_feature, vectorizer=vectorizer)
    control_ds_global.append(speaker_ds)            # B1 data
    control_val_ds_global.append(speaker_val_ds)    # B2 data
    control_test_ds_global.append(speaker_test_ds)  # B3 data


def pretrain_model(model, filename):
    """
    Trains the speech recognition model as a speaker independent system on the UASpeech control data.
    Training method:
        Model is trained using data from all the speakers (in the global SPEAKERS list), except for a single target
        speaker. The model is then evaluated on that target speaker. The target speaker for this will be CM06

        The model is trained using the B1 and B2 audio data from UASPEECH for the speakers used for training,
        with B3 data used as the validation data during training.

        The model is evaluated on all of the B1, B2 and B3 audio data (i.e. every audio file) for the target speaker.
        This data is not seen by the model during training, hence the model is evaluated as a speaker independent
        system.

    The final evaluated results are output into a file by this function
    ---
    :param model: Keras model
                        TransformerModel to train on control data
    :param filename: string
                        name of the file to write data into.
    :return:
    """
    idx_to_char = vectorizer.get_vocabulary()   # List of character tokens. Character index position corresponds to mapping
    # create file if it doesn't exist already
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write('New session starts here\n')

    # Access target speaker for validation
    i = 7
    # Set the test set as all the audio data from the target speaker
    evaluation_ds = control_ds_global[i].concatenate(control_val_ds_global[i]).concatenate(control_test_ds_global[i])
    # COMMENT THE LINE BELOW TO EVALUATE ON ALL TARGET SPEAKER DATA. In practice this should always be commented, but it acts as a significant time saver
    # evaluation_ds = evaluation_ds.shard(num_shards=6, index=0)

    speakers_test_ds = control_test_ds_global[0:i] + control_test_ds_global[i + 1:]   # B3 data
    speakers_train_ds = control_ds_global[0:i] + control_ds_global[i + 1:]            # B1 data
    speakers_val_ds = control_val_ds_global[0:i] + control_val_ds_global[i + 1:]    # B2 data

    # Getting the training dataset
    train_ds = speakers_train_ds[0].concatenate(speakers_val_ds[0])
    for j in range(1, len(speakers_train_ds)):
        train_ds = train_ds.concatenate(speakers_train_ds[j]).concatenate(speakers_val_ds[j])

    # Getting the validation dataset
    val_ds = speakers_test_ds[0]
    for k in range(1, len(speakers_test_ds)):
        val_ds = val_ds.concatenate(speakers_test_ds[k])

    batch = next(iter(val_ds.unbatch().batch(10)))

    # Fitting the model
    print(f'\n---------------------------------------------------------------------\n'
          f'Fitting model TransformerModel2.3 and evaluated for speaker {CONTROL_SPEAKERS[i]}\n'
          f'---------------------------------------------------------------------\n')
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                 SaveBestModel(set_model_to_best=True,
                               version_name=f'.3_Pretrained'),
                 DisplayOutputs(batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2)]
    # get best trained model
    with strategy.scope():
        history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
        test_loss = model.evaluate(evaluation_ds)                   # evaluate test loss
    wer = model.get_accuracy_score(evaluation_ds, metric="wer")     # evaluate WER
    loss = min(history.history['loss'])                             # evaluate training loss
    val_loss = min(history.history['val_loss'])                     # evaluate validation loss
    # write results to file
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write(
            f'Model: TransformerModel2.3, Speaker: {CONTROL_SPEAKERS[i]}, Loss: {loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, WER: {wer}, WRA: {1 - wer}\n')


def transfer_learning(model, filename, resume_training=0):
    """
    Trains the speech recognition model as a speaker dependent system on the UASpeech dystharic data, while implementing
    transfer learning.
    Training method:
        The model is first preloaded with weights from a previously trained model from the pretrain_model()
        Parts of the encoder input layer (SpeechFeatureEmbedding layer) is then frozen.
        The early encoder layers are also frozen to enable transfer learning.

        The model is then trained as a speaker dependent system (see more in ModelFit2.py)

    The final evaluated results are output into a file by this function
    ---
    :param model: Keras model
                        TransformerModel to train on control data
    :param filename: string
                        name of the file to write data into.
    :param resume_training: int
                               This functions trains and evaluates the models by taking the target speaker sequentially
                               from the global (CONTROL)SPEAKERS list, starting at index 'resume_training'.
    :return:
    """
    idx_to_char = vectorizer.get_vocabulary()  # List of character tokens. Character index position corresponds to mapping
    # create file if it doesn't exist already
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write('New session starts here\n')

    # Loop through every speaker, using the current speaker as the target speaker
    for i in range(resume_training, len(SPEAKERS)):
        # Getting the train, val, evaluation sets
        train_ds = ds_global[i].concatenate(val_ds_global[i])
        train_ds = train_ds.shuffle(buffer_size=100, reshuffle_each_iteration=False)
        train_ds = train_ds.skip(4)
        val_ds = train_ds.take(4)
        evaluation_ds = test_ds_global[i]
        batch = next(iter(val_ds.unbatch().batch(10)))

        print(f'\n---------------------------------------------------------------------\n'
              f'Fitting model TransformerModel2.3 and evaluated for speaker {SPEAKERS[i]}\n'
              f'---------------------------------------------------------------------\n')

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                     SaveBestModel(set_model_to_best=True,
                                   version_name='.3_' + SPEAKERS[i] + '_TransferLearning'),
                     DisplayOutputs(batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2)]
        # Fit model
        with strategy.scope():
            # quick model fit to get input shape for loading weights
            model.fit(val_ds.take(1), epochs=1, verbose=0)
            # load pre-trained model weights from the UASpeech control data
            model.load_weights('/home/ylei532/tmp/pycharm_project_921/Models/TransformerModel2.3_Pretrained.keras')

            # Freeze parts of the encoder input layer
            model.enc_input.conv1.trainable = SpeechEmbeddingConv1_trainable
            model.enc_input.conv2.trainable = SpeechEmbeddingConv2_trainable
            model.enc_input.conv3.trainable = SpeechEmbeddingConv3_trainable
            # Freeze the first few encoder layers
            for j in range(1, encoder_layers_to_freeze+1):
                model.encoder.layers[j].trainable = False

            history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
            test_loss = model.evaluate(evaluation_ds)                # evaluate test loss
        wer = model.get_accuracy_score(evaluation_ds, metric="wer")  # evaluate WER
        loss = min(history.history['loss'])                          # evaluate training loss
        val_loss = min(history.history['val_loss'])  # evaluate validation loss
        # write results to file
        with open(filename, mode="a", encoding="utf-8") as f:
            f.write(
                f'Model: TransformerModel2.3, Speaker: {SPEAKERS[i]}, Loss: {loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, WER: {wer}, WRA: {1 - wer}\n')


def main():
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    )
    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(control_ds_global[0]),
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model = Transformer(
        num_hid=320,
        num_head=8,
        num_feed_forward=192,
        target_maxlen=maxlen,
        num_layers_enc=5,
        num_layers_dec=2,
        num_classes=34,
        kernel_size=5,
        dropout_rate=0.5,
    )
    model.compile(optimizer=optimizer, loss=loss_fn)
    # pretrain_model(model, './Models/SpeakerIndependentTransformer2.3_CM06.txt')
    transfer_learning(model, filename="./Models/TransformerModel2.3TransferLearning.txt")


if __name__ == "__main__":
    main()






















