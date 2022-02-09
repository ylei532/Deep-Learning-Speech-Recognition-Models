from tensorflow import keras
from tensorflow.keras import layers
from TORGO_Preprocessing import *
from nltk.translate.bleu_score import sentence_bleu


'''''''''''''''''''''''
-----------------------
TRANSFORMER INPUT LAYER
-----------------------
'''''''''''''''''''''''


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.maxlen = maxlen
        self.num_vocab = num_vocab
        self.num_hid = num_hid
        self.emb = layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "num_vocab": self.num_vocab,
            "num_hid": self.num_hid
        })
        return config


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, kernel_size=11):
        super().__init__()
        self.num_hid = num_hid
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, kernel_size, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid,
            "kernel_size": self.kernel_size
        })
        return config


class GatedLinearUnit(layers.Layer):
    def __init__(self, num_hid):
        super(GatedLinearUnit, self).__init__()
        self.Linear = layers.Dense(num_hid)
        self.sigmoid = layers.Dense(num_hid, activation="sigmoid")

    def call(self, inputs):
        return self.Linear(inputs) * self.sigmoid(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid
        })
        return config


'''''''''''''''''''''''
-----------------------
  TRANSFORMER ENCODER
-----------------------
'''''''''''''''''''''''


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self.attn = GatedLinearUnit(embed_dim)
        self.leftConv1D = layers.Conv1D(self.embed_dim, 11, strides=1, padding="same", activation="relu")
        self.rightConv1D = layers.Conv1D(self.embed_dim, 11, strides=1, padding="same", activation="relu")
        self.SepConv1D = layers.SeparableConv1D(self.embed_dim, 11, padding="same")

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_input = self.layernorm1(inputs)
        attn_output = inputs + self.attn(attn_input)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm2(attn_output)
        ffn_output = self.leftConv1D(out1) + self.rightConv1D(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.layernorm3(ffn_output)
        return self.SepConv1D(ffn_output) + attn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim
        })
        return config


'''''''''''''''''''''''
-----------------------
TRANSFORMER DECODER
-----------------------
'''''''''''''''''''''''


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(dropout_rate)
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'feed_forward_dim': self.feed_forward_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


'''''''''''''''''''''''
-----------------------
  TRANSFORMER MODEL
-----------------------
'''''''''''''''''''''''


class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=34,
        kernel_size=11,
        dropout_rate=0.1
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.vectorizer = VectorizeChar(max_len=self.target_maxlen)
        self.idx_to_char = self.vectorizer.get_vocabulary()
        self.model_name = 'TransformerModel2'
        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, kernel_size=kernel_size)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input] +
            [
                TransformerEncoder(num_hid, num_head, num_feed_forward, dropout_rate)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward, dropout_rate),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        """Forward pass of model"""
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

    def predict_text(self, batch, print_mode=True, metric=None):
        """Performs the predictions of the words spoken in an audio file, and returns an accuracy metric value,
        either of 'bleu' or 'wer'"""
        assert (metric == "bleu" or metric == "wer" or metric is None)
        score = 0
        source = batch["source"]
        target = batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.generate(source, 1)
        preds = preds.numpy()

        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == 2:
                    break
            target_text = target_text.replace('-', '')
            if print_mode:
                print(f"target:     {target_text}")
                print(f"prediction: {prediction}\n")
            if metric == "bleu":
                target_text = [target_text[1:-1].split(' ')]
                prediction = prediction[1:-1].split(' ')
                n_grams = len(target_text[0])
                if n_grams == 1:
                    weights = (1, 0, 0, 0)
                elif n_grams == 2:
                    weights = (0.5, 0.5, 0, 0)
                elif n_grams == 3:
                    weights = (1 / 3, 1 / 3, 1 / 3, 0)
                else:
                    weights = (0.25, 0.25, 0.25, 0.25)

                score += sentence_bleu(target_text, prediction, weights=weights)
            elif metric == "wer":
                if target_text == prediction:
                    score += 1
            else:
                continue
        if metric is not None:
            if metric == "wer":
                print('{} score of one validation batch: {:.2f}\n'.format(metric.upper(), 1 - score / float(bs)))
            elif metric == "bleu":
                print('{} score of one validation batch: {:.2f}\n'.format(metric.upper(), score / float(bs)))
        return score, bs

    def get_accuracy_score(self, ds, metric="bleu", print_mode=False):
        """Get the accuracy score from a dataset. The possible metrics are: BLEU score and Word Error Rate"""
        score = 0
        samples = 0
        ds_itr = iter(ds)

        for batch in ds_itr:
            score_per_batch, bs = self.predict_text(batch, print_mode=print_mode, metric=metric)
            score += score_per_batch
            samples += bs

        if metric == "wer":
            print('Average {} score of ds: {:.2f}\n'.format(metric.upper(), 1 - (score / float(samples))))
            return 1 - (score / float(samples))
        print('Average {} score of ds: {:.2f}\n'.format(metric.upper(), score / float(samples)))
        return score / float(samples)


'''''''''''''''''''''''
-----------------------
    CUSTOM CALLBACKS
-----------------------
'''''''''''''''''''''''


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
            self, batch, idx_to_token, target_start_token_idx=1, target_end_token_idx=2
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token
        self.bleu_score = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        self.model.predict_text(self.batch, metric='bleu')


class SaveBestModel(keras.callbacks.Callback):
    def __init__(self, set_model_to_best=False, version_name=''):
        """ Saves best model weights to file
        Args:
            set_model_to_best: Boolean, True to set model to weights with the lowest validation loss during training (at the end of training)
            version_name: String to customize filename to save weights to
        """
        self.val_loss_start = 99999999999999999999.0
        self.val_loss_end = 0
        self.set_model_to_best = set_model_to_best
        self.version_name = version_name

    def on_epoch_end(self, epoch, logs=None):

        self.val_loss_end = logs.get('val_loss')
        if self.val_loss_end < self.val_loss_start:
            self.model.save_weights(
                f'/home/ylei532/tmp/pycharm_project_921/Models/{self.model.model_name + self.version_name}.keras')
            self.val_loss_start = self.val_loss_end

    def on_train_end(self, logs=None):
        if self.set_model_to_best:
            self.model.load_weights(
                f'/home/ylei532/tmp/pycharm_project_921/Models/{self.model.model_name + self.version_name}.keras')


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=85,
            steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
                self.init_lr + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup - (epoch - self.warmup_epochs) * (
                    self.lr_after_warmup - self.final_lr) / self.decay_epochs
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)
