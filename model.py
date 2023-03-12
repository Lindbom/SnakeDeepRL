
import tensorflow as tf


class model(tf.keras.Sequential):
     

    def __init__(self, input_shape) -> None:
        super().__init__([
        tf.keras.layers.Dense(12, activation = tf.nn.relu, input_shape = (input_shape, )), 
        tf.keras.layers.Dense(20, activation = tf.nn.relu), 
        tf.keras.layers.Dense(24, activation = tf.nn.relu), 
        tf.keras.layers.Dense(24, activation = tf.nn.relu)],
        )
        self.optimizer = tf.keras.optimizers.Adam(
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            weight_decay=None,
                            clipnorm=None,
                            clipvalue=None,
                            global_clipnorm=None,
                            use_ema=False,
                            ema_momentum=0.99,
                            ema_overwrite_frequency=None,
                            jit_compile=True,
                            name='Adam',
                        )

    def get_loss(self, logits, action, reward):
        logprob = tf.nn.softmax_cross_entropy_with_logits(
                logits = logits,
                labels = action,
                )
        print(logprob)

        # logprob = tf.reduce_mean(logprob)

    # def apply_gradient(self, model, optimizer):
    #     with tf.GradientTape as tape:
            