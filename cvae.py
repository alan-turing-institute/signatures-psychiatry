import tensorflow as tf
import numpy as np
from tqdm import tqdm

class CVAE:
    def __init__(self, n_latent=14, seed=None):
        self.n_latent = n_latent
        tf.set_random_seed(seed)
    


    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X_in, cond, input_dim):
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=None):
            x = tf.concat([X_in, cond], axis=1)                   ###
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=50)
            mn = tf.layers.dense(x, units=self.n_latent)
            sd       = 0.5 * tf.layers.dense(x, units=self.n_latent)            
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd))

            return z, mn, sd


    def decoder(self, sampled_z, cond, inputs_decoder, input_dim):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.concat([sampled_z, cond], axis=1)                            ###
            x = tf.layers.dense(x, units=inputs_decoder, activation=self.lrelu) ###
            x = tf.layers.dense(x, units=inputs_decoder, activation=self.lrelu)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=50)
            x = tf.layers.dense(x, units=input_dim, activation=tf.nn.sigmoid)
            return x


    def train(self, data, data_cond, n_epochs = 10000):
        tf.reset_default_graph()

        batch_size = 64
        input_dim = data.shape[1]
        dim_cond = data_cond.shape[1]
        
        X_in = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
        self.cond = tf.placeholder(dtype=tf.float32, shape=[None, dim_cond], name='c')
        Y = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Y')
        Y_flat = Y

        dec_in_channels = 1

        reshaped_dim = [-1, 7, 7, dec_in_channels]
        inputs_decoder = 49 * dec_in_channels // 2


        self.sampled, mn, sd = self.encoder(X_in, self.cond, input_dim=input_dim)

        self.dec = self.decoder(self.sampled, self.cond, inputs_decoder=inputs_decoder, input_dim=input_dim)

        unreshaped = tf.reshape(self.dec, [-1, input_dim])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)

        alpha = 0.02

        loss = tf.reduce_mean((1-alpha) * img_loss + alpha * latent_loss)

        optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        batch_size = len(data)
        n_batches = len(data) // batch_size

        for i in tqdm(range(n_epochs), desc="Training"):
            ii = i % n_batches
            batch = data[batch_size*ii:batch_size * (ii + 1)]
            batch_cond = data_cond[batch_size * ii:batch_size * (ii + 1)]
            self.sess.run(optimizer, feed_dict = {X_in: batch, self.cond: batch_cond, Y: batch})


    def generate(self, cond, n_samples=None):
        if n_samples == 0:
            return []
        
        if n_samples is not None:
            randoms = [np.random.normal(0, 1, self.n_latent) for _ in range(n_samples)]
            cond = [list(cond)] * n_samples
            
        else:
            randoms = [np.random.normal(0, 1, self.n_latent)]
            cond = [list(cond)]
        
        
        
        samples = self.sess.run(self.dec, feed_dict = {self.sampled: randoms, self.cond: cond})
        
        if n_samples is None:
            return samples[0]
        
        return samples

        

