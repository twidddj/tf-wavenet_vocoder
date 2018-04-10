import tensorflow as tf
from apps.vocoder.model.vocoder import Vocoder

def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-8, beta1=0.95, beta2=0.9995)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}