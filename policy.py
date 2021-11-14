import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

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

    def F(self, theta, gamma=0.99):
        # START HIDE
        done = False
        s = self.env.reset()
        total_reward = 0
        discount = 1
        while not done:
          a = self.get_action(s, theta)
          s, r, done, _ = self.env.step(a)
          total_reward += discount * r
          discount *= gamma
        # END HIDE
        return total_reward
    
    def evaluate(self, theta):
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
    def __init__(self, env, state_dim, nA, min_logvar=1, max_logvar=3):
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

    def F(self, theta, gamma=.99, max_step=5e3):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        discount = 1
        steps = 0
        # while not done:
        while not done and (steps < max_step):
            # WRITE CODE HERE
            fn = lambda a: [theta[2*a*(self.state_dim+1)] + state @ theta[2*a*(self.state_dim+1)+1: (2*a+1)*(self.state_dim+1)], 
                            theta[(2*a+1)*(self.state_dim+1)] + state @ theta[(2*a+1)*(self.state_dim+1)+1: (2*a+2)*(self.state_dim+1)]]
            mvs = np.array(list(map(fn, a_dim))).flatten()
            a_mean, a_v  = self.get_output(np.expand_dims(mvs, 0))
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, theta):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        steps = 0
        while not done:
            # WRITE CODE HERE
            fn = lambda a: [theta[2*a*(self.state_dim+1)] + state @ theta[2*a*(self.state_dim+1)+1: (2*a+1)*(self.state_dim+1)], 
                            theta[(2*a+1)*(self.state_dim+1)] + state @ theta[(2*a+1)*(self.state_dim+1)+1: (2*a+2)*(self.state_dim+1)]]
            mvs = np.array(list(map(fn, a_dim))).flatten()
            a_mean, a_v  = self.get_output(np.expand_dims(mvs, 0))
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward
            steps += 1
        print("The length of the trajectory is {}".format(steps))
        return G


    def nnF(self, nn, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        discount = 1
        steps = 0
        while not done:
        # while not done and (steps < max_step):
            a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G
    
    def nneval(self, nn, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        a_dim = np.arange(self.nA)
        while not done:
        # while not done and (steps < max_step):
            a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward

        return G

class GausNN(object):
    def __init__(self, env, nn, state_dim, nA, min_logvar=1, max_logvar=5):
        self.env = env
        self.nn = nn
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.nA = nA
        self.input_dim = state_dim

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        means = output[:, :self.nA]
        raw_vs = output[:, self.nA:]
        logvars = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_vs)
        logvars = self.min_logvar + tf.nn.softplus(logvars - self.min_logvar)
        return means, tf.exp(logvars).numpy()

    def F(self, theta, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        self.nn.update_params(self.nn.theta2nnparams(theta, self.input_dim, self.nn.output_dim))
        while not done:
        # while not done and (steps < max_step):
            a_mean, a_v  = self.get_output(self.nn(np.expand_dims(state, 0)).numpy())
            # action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, nn):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            a_mean, a_v  = self.get_output(nn(np.expand_dims(state, 0)).numpy())
            # action = np.tanh(np.random.normal(a_mean[0], a_v[0]))
            action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G

class Energy(object):
    def __init__(self, env, nn, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.nn = nn
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.input_dim = nA + state_dim
        self.nA=nA

    def energy_action(self, actor, state, K):
        sample_actions = np.random.uniform(low=-2.0, high=2.0, size=(K,self.nA))
        #states = np.repeat(state, K).reshape((K,state.size))#this gives a wrong matrix
        states = np.tile(state,(K,1))
        sas = np.concatenate((states, sample_actions), axis=1)
        energies = actor(sas).numpy().reshape(-1)
        return(sample_actions[np.argmin(energies)])  

    
    def F(self, theta, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        self.nn.update_params(self.nn.theta2nnparams(theta, self.input_dim, 1))
        while not done:
        # while not done and (steps < max_step):
            action = self.energy_action(self.nn, state, K=self.nA*10)
            # action = np.random.normal(a_mean[0], a_v[0])

            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, nn):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            action = self.energy_action(nn, state, K=self.nA*10)

            state, reward, done, _ = self.env.step(action)
            G += reward
        return G

class Energy_polyn(object):
    def __init__(self, env, state_dim, nA, min_logvar=1, max_logvar=3):
        self.env = env
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.input_dim = nA + state_dim
        self.nA=nA

    def energy_action(self, theta, state, K):    
        sample_actions = np.random.uniform(low=-2.0, high=2.0, size=(K,self.nA))
        #states = np.repeat(state, K).reshape((K,state.size))#this gives a wrong matrix
        states = np.tile(state,(K,1))
        sas = np.concatenate((states, sample_actions), axis=1)
        sas_Matrix=PolynomialFeatures(degree=2, include_bias=False).fit_transform(sas)
        energies=sas_Matrix@theta
        return(sample_actions[np.argmin(energies)])  
    
    def F(self, theta, gamma=.99, max_step=1e4):
        G = 0.0
        state = self.env.reset()
        done = False
        discount = 1
        steps = 0
        while not done:
        # while not done and (steps < max_step):
            action = self.energy_action(theta, state, K=self.nA*10)
            state, reward, done, _ = self.env.step(action)
            G += reward * discount
            discount *= gamma
            steps += 1
        return G

    def eval(self, theta):
        G = 0.0
        state = self.env.reset()
        done = False
        while not done:
            action = self.energy_action(theta, state, K=self.nA*10)
            state, reward, done, _ = self.env.step(action)
            G += reward
        return G


