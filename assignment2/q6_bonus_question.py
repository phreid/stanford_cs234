import gym
import tensorflow as tf
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN

from configs.q6_bonus_question import config

class DoubleQN(NatureQN):

    def add_loss_op(self, q, target_q):
        num_actions = self.env.action_space.n

        max_idx = tf.argmax(q, axis=1)
        idx = tf.stack([list(range(config.batch_size)), max_idx], axis=1)

        target = self.r + config.gamma * tf.gather_nd(target_q, idx)
        q_samp = tf.where(self.done_mask, self.r, target)
        one_hot = tf.one_hot(self.a, depth=num_actions, on_value=1.0, off_value=0.0)

        self.loss = tf.reduce_mean(
            tf.squared_difference(q_samp, tf.reduce_sum(q * one_hot, axis=1)))

class nature_config(config):
    output_path = config.output_path + "nature"

class double_config(config):
    output_path = config.output_path + "double"


if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
            overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
        config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
        config.lr_nsteps)

    # train model
    model_double = DoubleQN(env, double_config)
    model_double.run(exp_schedule, lr_schedule)

    tf.reset_default_graph()

    model = NatureQN(env, nature_config)
    model.run(exp_schedule, lr_schedule)