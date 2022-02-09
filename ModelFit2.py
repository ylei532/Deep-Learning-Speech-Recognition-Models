import keras.callbacks
import os
import keras_tuner as kt
from UASpeech_Preprocessing import *
# from TransformerModel import *
from TransformerModel2 import *
# from TransformerModel4 import *


"""
NOTES:
     - This file is used for training and evaluating the models (tuned from TORGO) on UASpeech dataset
     - Uncomment/comment imports to access the required DL model. ONLY have ONE model uncommented at a time. 
     - Models are described with different suffixes. The number to the left of the decimal point references the models
       architecture, and the number to the right of the decimal point distinguish model with different hyperparameters.
     - Models are saved under the following naming convention: TransformerModel{version}_{target speaker}_{system type}
       (where system type refers to one of speaker independent, speaker dependent or speaker adaptive)
     - Refer to notes in ModelFit.py file for more information on the models 
"""

strategy = tf.distribute.MultiWorkerMirroredStrategy()
maxlen = 50
vectorizer = VectorizeChar(maxlen)
SPEAKERS = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04', 'M05', 'M07', 'M08', 'M09', 'M12', 'M16']  # Recommended to read speaker_independent() function dosctring below before altering this list

ds_global = []        # Store B1 data from UASPEECH speakers
val_ds_global = []    # Store B2 data from UASPEECH speakers
test_ds_global = []   # store B3 data from UASPEECH speakers
for i in range(len(SPEAKERS)):
    speaker_ds, speaker_val_ds, speaker_test_ds = get_data_set_UA([SPEAKERS[i]], feature_extractor=audio_file_to_feature, vectorizer=vectorizer)
    ds_global.append(speaker_ds)            # B1 data
    val_ds_global.append(speaker_val_ds)    # B2 data
    test_ds_global.append(speaker_test_ds)  # B3 data


def build_model(hp):
    """
    Passed as a parameter to the KerasTuner search function
    ---
    :param hp: HyperParameters object.
                  see more at https://keras.io/api/keras_tuner/hyperparameters/
    :return: returns a compiled Transformer Model with randomly selected hyperparameters
    """
    # specify the hyperparameters for tuning, and the value range
    num_hid = hp.Int(name="num_hid", min_value=64, max_value=512, step=64)
    num_head = hp.Int(name="num_head", min_value=2, max_value=8, step=2)
    num_feed_forward = hp.Int(name="num_feed_forward", min_value=64, max_value=512, step=64)
    num_layers_enc = hp.Int(name="num_layers_enc", min_value=1, max_value=5, step=1)
    num_layers_dec = hp.Int(name="num_layers_dec", min_value=1, max_value=5, step=1)
    dropout_rate = hp.Float(name="dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    kernel_size = hp.Int(name="kernel_size", min_value=5, max_value=21, step=2)

    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=maxlen,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=34,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    )
    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(ds_global[0]),
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


def get_tuner(project_name, directory='/home/ylei532/tmp/pycharm_project_921/Models'):
    """
    Retrieve tuner object with search logs from a previous search
    :param project_name: string
                            name of file that contains keras tuner search logs
    :param directory: string
                         name of directory that contains keras tuner search logs
    :return: keras tuner object
    """
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=100,
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        overwrite=False
    )

    return tuner


def speaker_independent(best_hps, filename, resume_training=0):
    """
    Trains and evaluates the speech recognition model as a speaker independent system.
    Training method:
        Model is trained using data from all the speakers (in the global SPEAKERS list), except for a single target
        speaker. The model is then evaluated on that target speaker.

        The model is trained using the B1 and B2 audio data from UASPEECH for the speakers used for training,
        with B3 data used as the validation data during training.

        The model is evaluated on all of the B1, B2 and B3 audio data (i.e. every audio file) for the target speaker.
        This data is not seen by the model during training, hence the model is evaluated as a speaker independent
        system.

    The final evaluated results are output into a file by this function
    ---
    :param best_hps: int
                        Number of hyperparameter configurations for a model to evaluate for. Max is 10
    :param filename: string
                        name of the file to write data into.
    :param resume_training: int
                               This functions trains and evaluates the models by taking the target speaker sequentially
                               from the global SPEAKERS list, starting at index 'resume_training'.
    :return:
    """
    idx_to_char = vectorizer.get_vocabulary()   # List of character tokens. Character index position corresponds to mapping
    # create file if it doesn't exist already
    with open(filename, mode="a", encoding="utf-8") as f:
        f.write('New session starts here\n')

    # Loop through every speaker, using the current speaker as the target speaker
    for i in range(resume_training, len(SPEAKERS)):
        # Set the test set as all the audio data from the target speaker
        evaluation_ds = ds_global[i].concatenate(val_ds_global[i]).concatenate(test_ds_global[i])
        evaluation_ds = evaluation_ds.shard(num_shards=6, index=0)  # COMMENT THIS LINE TO EVALUATE ON ALL TARGET SPEAKER DATA. The random filtering of data here is bad practice, but need to save time...

        speakers_test_ds = test_ds_global[0:i] + test_ds_global[i+1:]   # B3 data
        speakers_train_ds = ds_global[0:i] + ds_global[i+1:]            # B1 data
        speakers_val_ds = val_ds_global[0:i] + val_ds_global[i+1:]    # B2 data

        # Getting the training dataset
        train_ds = speakers_train_ds[0].concatenate(speakers_val_ds[0])
        for j in range(1, len(speakers_train_ds)):
            train_ds = train_ds.concatenate(speakers_train_ds[j]).concatenate(speakers_val_ds[j])

        # Getting the validation dataset
        val_ds = speakers_test_ds[0]
        for k in range(1, len(speakers_test_ds)):
            val_ds = val_ds.concatenate(speakers_test_ds[k])

        batch = next(iter(val_ds.unbatch().batch(10)))
        # model version tracking
        version = 1
        # Fit the model to each hyperparameter configuration passed in the input
        for hp in best_hps:
            print(f'\n---------------------------------------------------------------------\n'
                  f'Fitting model version {str(version)} and evaluated for speaker {SPEAKERS[i]}\n'
                  f'---------------------------------------------------------------------\n')
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                         SaveBestModel(set_model_to_best=True,
                                       version_name='.' + str(version) + '_' + SPEAKERS[i] + '_Independent'),
                         DisplayOutputs(batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2)]
            # get best trained model
            with strategy.scope():
                model = build_model(hp)
                history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
                test_loss = model.evaluate(evaluation_ds)                   # evaluate test loss
            wer = model.get_accuracy_score(evaluation_ds, metric="wer")     # evaluate WER
            loss = min(history.history['loss'])                             # evaluate training loss
            val_loss = min(history.history['val_loss'])                     # evaluate validation loss
            # write results to file
            with open(filename, mode="a", encoding="utf-8") as f:
                f.write(
                    f'Model: {model.model_name + "." + str(version)}, Speaker: {SPEAKERS[i]}, Loss: {loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, WER: {wer}, WRA: {1 - wer}\n')
            version += 1


def speaker_dependent(best_hps, filename, resume_training=0):
    """
    Trains and evaluates the speech recognition model as a speaker dependent system
    Training method:
        Model is trained on a single target speaker at any one time. The model is trained using the B1 and B2 audio data
        of the target speaker (with some of the data taken out for validation). The model is evaluated on the B3 audio
        data of the target speaker.

    The final evaluated results are output into a file by this function
    ---
    :param best_hps: int
                        Number of hyperparameter configurations for a model to evaluate for. Max is 10
    :param filename: string
                        name of the file to write data into.
    :param resume_training: int
                               This functions trains and evaluates the models by taking the target speaker sequentially
                               from the global SPEAKERS list, starting at index 'resume_training'.
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
        # model version tracking
        version = 1

        # Fit the model to each hyperparameter configuration passed in the input
        for hp in best_hps:
            print(f'\n---------------------------------------------------------------------\n'
                  f'Fitting model version {str(version)} and evaluated for speaker {SPEAKERS[i]}\n'
                  f'---------------------------------------------------------------------\n')
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                         SaveBestModel(set_model_to_best=True,
                                       version_name='.'+str(version) + '_' + SPEAKERS[i] + '_Dependent'),
                         DisplayOutputs(batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2)]
            # get best trained model
            with strategy.scope():
                model = build_model(hp)
                history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
                test_loss = model.evaluate(evaluation_ds)                                    # evaluate test loss
            wer = model.get_accuracy_score(evaluation_ds, metric="wer")     # evaluate WER
            loss = min(history.history['loss'])                                              # evaluate training loss
            val_loss = min(history.history['val_loss'])                                      # evaluate validation loss
            # write results to file
            with open(filename, mode="a", encoding="utf-8") as f:
                f.write(
                    f'Model: {model.model_name + "." + str(version)}, Speaker: {SPEAKERS[i]}, Loss: {loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, WER: {wer}, WRA: {1-wer}\n')
            version += 1


def speaker_adaptive(best_hps, filename, resume_training=0):
    """
    Trains and evaluates the speech recognition model as a speaker adaptive system
    Training method:
        Model is FIRST trained using data from all the speakers (in the global SPEAKERS list), except for a single target
        speaker. The model is then trained again on that target speaker to improve its ability to recognise the specific speaker.

        In this implementation, the model is first 'trained' from all the speakers (except for the target speaker)
        by loading the model weights from the models trained and saved from the speaker_independent() function. The
        model is then trained and evaluated in the same way as explained in the speaker_dependent docstring

    The final evaluated results are output into a file by this function
    ---
    :param best_hps: int
                        Number of hyperparameter configurations for a model to evaluate for. Max is 10
    :param filename: string
                        name of the file to write data into.
    :param resume_training: int
                               This functions trains and evaluates the models by taking the target speaker sequentially
                               from the global SPEAKERS list, starting at index 'resume_training'.
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
        # model version tracking
        version = 1

        # Fit the model to each hyperparameter configuration passed in the input
        for hp in best_hps:
            print(f'\n---------------------------------------------------------------------\n'
                  f'Fitting model version {str(version)} and evaluated for speaker {SPEAKERS[i]}\n'
                  f'---------------------------------------------------------------------\n')
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5),
                         SaveBestModel(set_model_to_best=True,
                                       version_name='.' + str(version) + '_' + SPEAKERS[i] + '_Adaptive'),
                         DisplayOutputs(batch, idx_to_char, target_start_token_idx=1, target_end_token_idx=2)]
            # get best trained model
            with strategy.scope():
                model = build_model(hp)
                model.fit(val_ds, epochs=1, verbose=0)
                model.load_weights(f'/home/ylei532/tmp/pycharm_project_921/Models/{model.model_name+ "." + str(version)+"_"+SPEAKERS[i]+"_Independent"}')
                history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
                test_loss = model.evaluate(evaluation_ds)
            wer = model.get_accuracy_score(evaluation_ds, metric="wer")
            loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            # write results to file
            with open(filename, mode="a", encoding="utf-8") as f:
                f.write(
                    f'Model: {model.model_name + "." + str(version)}, Speaker: {SPEAKERS[i]}, Loss: {loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}, WER: {wer}, WRA: {1 - wer}\n')
            version += 1


def main():
    model = "TransformerModel_Bayesian"
    tuner = get_tuner(project_name=model)
    best_hps = tuner.get_best_hyperparameters(3)

    # speaker_independent(best_hps, filename="./Models/SpeakerIndependentTransformer1.txt")
    # speaker_independent(best_hps, filename="SpeakerIndependentTransformer2.txt")
    # speaker_independent(best_hps, filename="SpeakerIndependentTransformer4.txt")

    # speaker_dependent(best_hps, filename="./Models/SpeakerDependentTransformer1.txt", resume_training=5)
    # speaker_dependent(best_hps, filename="./Models/SpeakerDependentTransformer2.txt", resume_training=11)
    # speaker_dependent(best_hps, filename="./Models/SpeakerDependentTransformer4.txt")

    # speaker_adaptive(best_hps, filename="SpeakerAdaptiveTransformer1.txt")
    # speaker_adaptive(best_hps, filename="SpeakerAdaptiveTransformer2.txt")
    # speaker_adaptive(best_hps, filename="SpeakerAdaptiveTransformer4.txt")


if __name__ == "__main__":
    main()

