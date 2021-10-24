import numpy as np
import tensorflow as tf


class Log(object):  
    def __init__(self, env):
        self.env = env

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_action(self, s, theta):
        w = theta[:4]
        b = theta[4]
        p_left = self.sigmoid(w @ s + b)
        a = np.random.choice(2, p=[p_left, 1 - p_left])
        return a

    def rl_fn(self, theta):
        # START HIDE
        done = False
        s = self.env.reset()
        total_reward = 0
        while not done:
          a = self.get_action(s, theta)
          s, r, done, _ = self.env.step(a)
          total_reward += r
        # END HIDE
        return total_reward

class Gaus(object):
    def __init__(self, env, state_dim, min_logvar=1, max_logvar=5, nA=1):
        self.env = env
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.nA = nA
        self.state_dim = state_dim


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        means = output[:, 0:self.nA]
        raw_vs = output[:, self.nA:]
        logvars = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_vs)
        logvars = self.min_logvar + tf.nn.softplus(logvars - self.min_logvar)
        return means, tf.exp(logvars).numpy()
    
    def rl_fn(self, theta):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            # WRITE CODE HERE
            mv = np.array([theta[0] + state @ theta[1:self.state_dim+1], 
                          theta[self.state_dim+1] + state @ theta[self.state_dim+2:]])
            a_mean, a_v  = self.get_output(np.expand_dims(mv, 0))
            action = np.random.multivariate_normal(a_mean[0], np.diag(a_v[0]))

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G
        




