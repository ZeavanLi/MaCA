import numpy as np
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
        # img=tf.convert_to_tensor(img)
        # info=tf.convert_to_tensor(info)
        img_feature = self.conv1(img)
        img_feature = self.pool1(img_feature)
        img_feature = self.conv2(img_feature)
        img_feature = self.pool2(img_feature)
        info_feature = self.info_fc(info)
        # print(img_feature.shape)
        # print(info_feature.shape)
        # combined = tf.concat([tf.reshape(img_feature, (1, tf.size(img_feature))), tf.reshape(info_feature, (1, tf.size(info_feature)))],axis=1)
        combined = tf.concat(
            [tf.reshape(img_feature, (img_feature.shape[0], -1)),
             tf.reshape(info_feature, (info_feature.shape[0], -1))],
            axis=1)
        # print('--------------------')
        # print(tf.reshape(img_feature, (img_feature.shape[0], -1)).shape)
        # print(tf.reshape(info_feature, (img_feature.shape[0], -1)).shape)
        # print(combined.shape)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action


def sync(net, net_tar):
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


class RLFighter:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_delay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_delay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.s_screen_memory = []
        self.s_info_memory = []
        self.a_memory = []
        self.r_memory = []
        self.s__screen_memory = []
        self.s__info_memory = []
        self.memory_counter = 0

        self.gpu_enable = tf.test.is_gpu_available()

        self.learn_step_counter = 0
        self.cost_his = []
        self.eval_net, self.target_net = NetFighter(self.n_actions), NetFighter(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!')
        self.loss_func = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

    def store_transition(self, s, a, r, s_):
        self.s_screen_memory.append(s['screen'])
        self.s_info_memory.append(s['info'])
        self.a_memory.append(a)
        self.r_memory.append(r)
        self.s__screen_memory.append(s_['screen'])
        self.s__info_memory.append(s_['info'])
        self.memory_counter += 1

    def __clear_memory(self):
        self.s_screen_memory.clear()
        self.s_info_memory.clear()
        self.a_memory.clear()
        self.r_memory.clear()
        self.s__screen_memory.clear()
        self.s__info_memory.clear()
        self.memory_counter = 0

    def choose_action(self, img_obs, info_obs):
        img_obs = tf.expand_dims(tf.convert_to_tensor(img_obs, dtype=tf.float32), 0)
        info_obs = tf.expand_dims(tf.convert_to_tensor(info_obs, dtype=tf.float32), 0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(img_obs, info_obs)
            action = tf.argmax(actions_value, 1)
            action = action.numpy()
        else:
            action = np.zeros(1, dtype=np.int32)
            action[0] = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            sync(self.eval_net, self.target_net)
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            checkpoint = tf.train.Checkpoint(myModel=self.target_net)
            manager = tf.train.CheckpointManager(checkpoint, directory='./model/tf2_DQN_model', max_to_keep=3)
            path = manager.save(checkpoint_number=None)
            print("model saved to %s" % path)
        s_screen_mem = tf.convert_to_tensor(np.array(self.s_screen_memory), dtype=tf.float32)
        s_info_mem = tf.convert_to_tensor(np.array(self.s_info_memory), dtype=tf.float32)
        a_mem = tf.convert_to_tensor(np.array(self.a_memory), dtype=tf.int32)
        r_mem = tf.convert_to_tensor(np.array(self.r_memory), dtype=tf.float32)
        r_mem = tf.reshape(r_mem, shape=(self.memory_counter, 1))
        s__screen_mem = tf.convert_to_tensor(np.array(self.s__screen_memory), dtype=tf.float32)
        s__info_mem = tf.convert_to_tensor(np.array(self.s__info_memory), dtype=tf.float32)

        print(a_mem)
        print(self.eval_net(s_screen_mem, s_info_mem).shape)
        print(a_mem.shape)

        with tf.GradientTape() as tape:
            q_eval = tf.gather_nd(self.eval_net(s_screen_mem, s_info_mem), a_mem, 1)
            print('q_eval.shape: {}'.format(q_eval.shape))
            q_next = self.target_net(s__screen_mem, s__info_mem)
            print(q_next.dtype)

            print(r_mem.dtype)
            print(tf.reshape(tf.argmax(q_next, 1), shape=(self.memory_counter, 1)).dtype)
            print(tf.cast(tf.reshape(tf.argmax(q_next, 1), shape=(self.memory_counter, 1)), dtype=tf.float32).dtype)

            q_target = r_mem + self.gamma * tf.cast(tf.reshape(tf.argmax(q_next, 1), shape=(self.memory_counter, 1)),
                                                    dtype=tf.float32)

            print(q_eval.shape)
            print(q_target.shape)

            loss = self.loss_func(q_eval, q_target)
        grads = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.eval_net.trainable_variables))

        # self.eval_net.compile(optimizer=self.optimizer, loss=loss,metrics=[tf.keras.metrics.sparse_categorical_crossentropy_accuracy])
        self.cost_his.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self.__clear_memory()


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
