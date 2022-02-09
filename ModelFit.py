import keras.callbacks
import keras_tuner as kt
# from TransformerModel import *
from TransformerModel2 import *
# from TransformerModel3 import *
# from TransformerModel4 import *
# from TransformerModel5 import *
# from TranformerModelFreeze import *
"""
NOTES:
     - This file is used for training and finetuning the models on the TORGO dataset to find the best hyperparameter
       configurations  
     - Parameters for feature extraction can be adjusted in the global variables in TORGO_Preprocessing.py file
     - Uncomment/comment imports to access the required DL model. ONLY have ONE model uncommented at a time
     - Models are described with different suffixes. The number to the left of the decimal point references the models
       architecture, and the number to the right of the decimal point distinguish model with different hyperparameters.
     - TransformerModel (model_name = TransformerModel1) uses the encoder-decoder transformer architecture from 
       the Deep Learning with Python book
     - TransformerModel2 uses a GatedLinearUnit instead of MHA for attention filtering in the encoder layer. This is
       based on the following article: 
       https://indico2.conference4me.psnc.pl/event/35/contributions/3122/attachments/301/324/Tue-1-8-5.pdf
     - TransformerModel3 uses an encoder-decoder architecture based on the following article:
       https://indico2.conference4me.psnc.pl/event/35/contributions/3122/attachments/301/324/Tue-1-8-5.pdf. This model
       provides excellent returns on the training and validation loss, however its text predictions are nonsense
     - TransformerModel4 uses an encoder architecture based off the one in TransformerModel1, but uses Conv1D layers 
       in the feed-forward layer
     - TransformerModel5 is based off TransformerModel4, but it also uses Conv1D layers in the feed-forward layer for 
       the decoder. Like TransformerModel3, this model also provides excellent returns on the training and validation
       loss, but its text predictions are nonsense
"""

strategy = tf.distribute.MultiWorkerMirroredStrategy()                  # used to enable data parallelism during training
maxlen = 200                                                            # set max length for text
vectorizer = VectorizeChar(maxlen)
SPEAKERS = ['F01', 'F03', 'F04', 'M01', 'M04', 'M02', 'M05']     # specify speakers to create ds from for model fitting
ds, val_ds, test_ds = get_data_set_TORGO(SPEAKERS, feature_extractor=audio_file_to_feature, vectorizer=vectorizer)


def model_fit(distributed_training=False, version_name="", existing_model_name=""):
    """
    Fit Transformer Model to TORGO dataset.
    Training method:
        The model is trained from all the audio files from all of the speakers. This data has been randomly shuffled
        adn split into training, validation and test data sets
    ___
    :param distributed_training: boolean
                                    Set to true to enable data parallelism during training
    :param  version_name: string
                             saves the weights of the model with the version_name as the suffix of the .keras file name
    :param  existing_model_name: string
                                    specify exising name of file to load weights from
    :return: returns the fitted model, and the object returned by model.fit()
    """
    idx_to_char = vectorizer.get_vocabulary()   # initialize vocabulary
    batch = next(iter(val_ds))                  # intialize a single validation batch
    # initialize callbacks for model.fit()
    display_cb = DisplayOutputs(batch,
                                idx_to_char,
                                target_start_token_idx=1,
                                target_end_token_idx=2
                                )
    model_checkpoint = SaveBestModel(set_model_to_best=True, version_name=version_name)
    # initialize loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    )
    # initialize learning rate schedule
    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(ds),
    )
    # initialize optimizer
    optimizer = keras.optimizers.Adam(learning_rate)

    if distributed_training:
        # Enabling data parallelism for training
        with strategy.scope():
            # construct model
            model = Transformer(
                num_hid=128,
                num_head=2,
                num_feed_forward=400,
                target_maxlen=maxlen,
                num_layers_enc=4,
                num_layers_dec=1,
                num_classes=34,
                dropout_rate=0.1,
                kernel_size=11,
            )
            # compile model
            model.compile(optimizer=optimizer, loss=loss_fn)
            if existing_model_name != "":
                # load weights to model from the specified file
                model.fit(ds, validation_data=val_ds, epochs=1, verbose=0)
                model.load_weights(f"/home/ylei532/tmp/pycharm_project_921/Models/{existing_model_name}")

            history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb, model_checkpoint], epochs=100)
    else:
        # constructing, compiling and fitting model without the use of data parallelism
        model = Transformer(
            num_hid=128,
            num_head=2,
            num_feed_forward=256,
            target_maxlen=maxlen,
            num_layers_enc=3,
            num_layers_dec=1,
            num_classes=34,
            kernel_size=11,
            dropout_rate=0.1,
        )
        model.compile(optimizer=optimizer, loss=loss_fn)
        if existing_model_name != "":
            model.fit(ds, validation_data=val_ds, epochs=1, verbose=0)
            model.load_weights(f"/home/ylei532/tmp/pycharm_project_921/Models/{existing_model_name}.keras")
        history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb, model_checkpoint], epochs=100)
        print("\n EVALUATING MODEL\n")
        model.evaluate(test_ds)
    return model, history


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
        steps_per_epoch=len(ds),
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


def get_best_parameters(project_name='untitled_project', search_method='BayesianOptimization', overwrite=False):
    """
    Uses KerasTuner to get the best hyperparameters for a model architecture
    ---
    :param project_name: string
                            name of the project to load from / save to. For best practice, set as the name of the model being tuned
    :param search_method: string
                             Set as 'BayesianOptimization' or 'Hyperband' to determine the search method.
    :param overwrite: boolean
                         Set to true to override the results from a previous search if it has a matching project_name
                         and exists in the same directory
    :return: Returns the best top_n hyperparamter combinations for the model, and the tuner object.
    """
    # initialize early stopping callback during the search
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    with strategy.scope():
        # creating the keras tuner object
        print(search_method == 'BayesianOptimization')
        if search_method == 'BayesianOptimization':
            tuner = kt.BayesianOptimization(
                build_model,
                objective="val_loss",
                max_trials=100,
                executions_per_trial=1,
                directory='/home/ylei532/tmp/pycharm_project_921/Models',
                project_name=project_name + '_Bayesian',
                overwrite=overwrite
            )
        elif search_method == 'Hyperband':
            tuner = kt.Hyperband(
                build_model,
                objective="val_loss",
                max_epochs=100,
                executions_per_trial=1,
                directory='/home/ylei532/tmp/pycharm_project_921/Models',
                project_name=project_name + '_Hyperband',
                overwrite=overwrite
            )
        # run search for best hyperparameters
        tuner.search(
            ds,
            batch_size=64,
            epochs=100,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )


def get_best_trained_model(hp, version):
    """
    Get the best model (lowest validation loss) with the specified Hyperparamter configuration
    ---
    :param hp: Hyperparameter object
                  specifies a hyperparameter configuration for the model
    :param version: string
                       saves the weights of the model with the version_name as the suffix of the .keras file name
    :return: returns best fitted model for given hp configuration
    """
    model = build_model(hp)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 SaveBestModel(set_model_to_best=True, name=version)]
    model.fit(ds, validation_data=val_ds, callbacks=callbacks, epochs=150)

    return model


def main():
    model_fit()
    # get_best_parameters(project_name='TransformerModel4')


if __name__ == "__main__":
    main()
