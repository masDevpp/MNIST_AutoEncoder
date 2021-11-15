import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, model_index, input_shape=(28, 28, 1), lr=0.001):
        self.input_shape = input_shape
        self.lr = lr

        self.models = [
            self.build_model_0,
            self.build_model_1,
            self.build_model_2,
            self.build_model_3,
            self.build_model_4,
            self.build_model_5,
            self.build_model_6,
            self.build_model_7,
            self.build_model_8,
            self.build_model_9,
            self.build_model_10,
            self.build_model_11,
            self.build_model_12,
            self.build_model_13,
        ]

        self.model, self.mod_support = self.models[model_index](self.input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model_0(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_1(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_2(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_3(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                            # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                         # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                        # 14x14x16
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(1, 3, use_bias=False, padding="same"),  # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_4(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                            # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(16, 3, use_bias=False, padding="same"), # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                         # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                        # 14x14x16
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(8, 3, use_bias=False, padding="same"),  # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(4, 3, use_bias=False, padding="same"),  # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SeparableConv2D(1, 3, use_bias=False, padding="same"),  # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_5(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        # Mask top 12 output
        v, _ = tf.math.top_k(encoder_out, 12)
        threshold = v[:, -1]
        mask = encoder_out >= tf.reshape(threshold, [-1,1])
        mask = tf.cast(mask, tf.float32)
        masked_out = encoder_out * mask

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(masked_out)

        return tf.keras.Model(inputs=input, outputs=[masked_out, decoder_out]), 0

    def build_model_6(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_7(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

        encoder_out = encoder_net(input)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(encoder_out)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0
    
    def build_model_8(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
        ])

        encoder_out = encoder_net(input)

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(activation)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_9(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2, use_bias=False),
        ])

        encoder_out = encoder_net(input)

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(activation)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_10(self, input_shape):
        input = tf.keras.Input(input_shape)

        encoder_conv_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
        ])

        encoder_conv_out = encoder_conv_net(input)

        input_mod = tf.keras.Input(1)

        encoder_dence_net = tf.keras.Sequential([
            tf.keras.layers.Dense(65, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2, use_bias=False),
        ])

        encoder_out = encoder_dence_net(tf.concat([encoder_conv_out, input_mod], 1))

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_dence_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(145, use_bias=False),
        ])

        decoder_dence_out = decoder_dence_net(activation)
        output_mod = decoder_dence_out[:,-1]

        decoder_dence_activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(decoder_dence_out[:,:-1])

        decoder_conv_net = tf.keras.Sequential([
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_conv_net(decoder_dence_activation)

        return tf.keras.Model(inputs=[input, input_mod], outputs=[encoder_out, decoder_out, output_mod]), 1

    def build_model_11(self, input_shape):
        # model 9 base, output 3 dim encoded vector
        input = tf.keras.Input(input_shape)

        encoder_net =  tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(3, use_bias=False),
        ])

        encoder_out = encoder_net(input)

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(144, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_net(activation)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out]), 0

    def build_model_12(self, input_shape):
        # model 10 base
        input = tf.keras.Input(input_shape)

        encoder_conv_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
        ])

        encoder_conv_out = encoder_conv_net(input)

        input_mod = tf.keras.Input(1)

        encoder_dence_net = tf.keras.Sequential([
            tf.keras.layers.Dense(65, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(3, use_bias=False),
        ])

        encoder_out = encoder_dence_net(tf.concat([encoder_conv_out, input_mod], 1))

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_dence_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(145, use_bias=False),
        ])

        decoder_dence_out = decoder_dence_net(activation)
        output_mod = decoder_dence_out[:,-1]

        decoder_dence_activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(decoder_dence_out[:,:-1])

        decoder_conv_net = tf.keras.Sequential([
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_conv_net(decoder_dence_activation)

        return tf.keras.Model(inputs=[input, input_mod], outputs=[encoder_out, decoder_out, output_mod]), 1

    def build_model_13(self, input_shape):
        # model 10 base
        input = tf.keras.Input(input_shape)

        encoder_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 7x7x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),                                    # 3x3x16 (=144)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(3, use_bias=False),
        ])

        encoder_out = encoder_net(input)

        activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(encoder_out)

        decoder_dence_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(8, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(145, use_bias=False),
        ])

        decoder_dence_out = decoder_dence_net(activation)
        output_mod = decoder_dence_out[:,-1]

        decoder_dence_activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])(decoder_dence_out[:,:-1])

        decoder_conv_net = tf.keras.Sequential([
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding="same"),  # 6x6x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),                                 # 12x12x16
            tf.keras.layers.ZeroPadding2D(),                                # 14x14x16
            tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="same"),   # 14x14x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="same"),   # 28x28x4
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1, 3, use_bias=False, padding="same"),   # 28x28x1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        decoder_out = decoder_conv_net(decoder_dence_activation)

        return tf.keras.Model(inputs=input, outputs=[encoder_out, decoder_out, output_mod]), 2

    def get_model_description(self):
        desc = ""

        for l in self.model.layers:
            if "sequential" in l.name:
                for ll in l.layers:
                    desc += str(ll.output_shape) + ", " + ll.name + "\n"
            else:
                desc += str(l.output_shape) + ", " + l.name + "\n"
        
        return desc

    def save_model(self, save_path):
        self.model.save_weights(save_path)

    def load_model(self, load_path):
        self.model.load_weights(load_path)

    @tf.function()
    def predict(self, x, mod=None, training=False):
        if self.mod_support == 0 or self.mod_support == 2:
            return self.model(x, training=training)
        elif self.mod_support == 1:
            return self.model([x, mod], training=training)

    @tf.function()
    def get_loss(self, x, y, x_mod=None, y_mod=None):
        loss = tf.keras.losses.MeanSquaredError()(x, y) / 2

        if self.mod_support == 1 or self.mod_support == 2:
            loss += tf.keras.losses.MeanSquaredError()(x_mod, y_mod) / 2
            
        return loss

    @tf.function()
    def train(self, x, mod=None):
        with tf.GradientTape() as tape:
            if self.mod_support == 0:
                # No mod input/output
                enc, pred =  self.model(x, training=True)
                loss = self.get_loss(x, pred)
            elif self.mod_support == 1:
                # Input x and mod then output pred and pred_mod
                enc, pred, pred_mod =  self.model([x, mod], training=True)
                loss = self.get_loss(x, pred, x_mod=mod, y_mod=pred_mod)
            elif self.mod_support == 2:
                # Input x only then output pred and pred_mod
                enc, pred, pred_mod =  self.model(x, training=True)
                loss = self.get_loss(x, pred, x_mod=mod, y_mod=pred_mod)
        
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return loss
