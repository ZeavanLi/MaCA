import tensorflow as tf


class NetFighter(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[5, 5],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            data_format='channels_first',
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, data_format='channels_first')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            data_format='channels_first',
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, data_format='channels_first')

        self.info_fc = tf.keras.layers.Dense(
            units=256,
            activation=tf.nn.tanh,
        )
        self.feature_fc = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )
        self.decision_fc = tf.keras.layers.Dense(units=n_actions)

    def call(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.pool1(img_feature)
        img_feature = self.conv2(img_feature)
        img_feature = self.pool2(img_feature)
        info_feature = self.info_fc(info)
        combined = tf.concat(
            [tf.reshape(img_feature, (img_feature.shape[0], -1)),
             tf.reshape(info_feature, (info_feature.shape[0], -1))],
            axis=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action


class RLFighter:
    def __init__(
            self,
            n_actions,
    ):
        self.n_actions = n_actions
        self.gpu_enable = tf.test.is_gpu_available()

        self.learn_step_counter = 0

        self.target_net = NetFighter(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!')
        #model_to_be_restored = NetFighter(self.n_actions)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=self.target_net)
        checkpoint.restore(tf.train.latest_checkpoint('./model/tf2_DQN_model'))

    def choose_action(self, img_obs, info_obs):
        img_obs = tf.expand_dims(tf.convert_to_tensor(img_obs, dtype=tf.float32), 0)
        info_obs = tf.expand_dims(tf.convert_to_tensor(info_obs, dtype=tf.float32), 0)
        actions_value = self.target_net(img_obs, info_obs)
        action = tf.argmax(actions_value, 1)
        action = action.numpy()
        print('action {}'.format(action))
        return action


class NetDetector(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            input=3,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            data_format='channels_first',
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, data_format='channels_first')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding='SAME',
            activation=tf.nn.relu,
            data_format='channels_first',
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, data_format='channels_first')

        self.info_fc = tf.keras.layers.Dense(
            units=256,
            activation=tf.nn.tanh
        )
        self.feature_fc = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )
        self.decision_fc = tf.keras.layers.Dense(units=n_actions)

    def call(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.pool1(img_feature)
        img_feature = self.conv2(img_feature)
        img_feature = self.pool2(img_feature)
        info_feature = self.info_fc(info)
        combined = tf.concat(
            [tf.reshape(img_feature, (img_feature.shape[0], -1)),
             tf.reshape(info_feature, (info_feature.shape[0], -1))],
            axis=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action
