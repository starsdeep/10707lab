import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class RBM(object):

    def __init__(self, k=1, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-1, reg=0.0, rng=None):

        self.params = {}
        self.reg = reg
        self.k = k
        if rng is None:
            self.rng = np.random.RandomState(1234)

        W = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        h_bias = np.zeros(hidden_dim)
        v_bias = np.zeros(input_dim)

        self.params = {'W': W, 'v_bias': v_bias, 'h_bias': h_bias}

    def loss(self, X, y=None):
        loss, grads = 0, {}
        loss = self.reconstruct_error(X)
        grads['W'], grads['v_bias'], grads['h_bias'] = self.CD_k(X)
        #weight decay
        grads['W'] += self.reg * self.params['W'] 
        return loss, grads

    def CD_k(self, v):
        #positive phase
        h_prob, h_sample = self.sample_h_given_v(v)
        positive_h_prob = np.copy(h_prob)
        #negative phase
        for i in range(self.k):
            v_prob, v_sample = self.sample_v_given_h(h_sample)
            h_prob, h_sample = self.sample_h_given_v(v_sample)
            
        dW = -(np.dot(v.T, positive_h_prob) - np.dot(v_sample.T, h_prob))
        dv_bias = -np.mean(v - v_sample, axis=0)
        dh_bias = -np.mean(positive_h_prob - h_prob, axis=0)
        return (dW, dv_bias, dh_bias)


    def sample_h_given_v(self,  v):
        h_prob = sigmoid(self.params['h_bias'] + np.dot(v, self.params['W']))
        h_sample = self.rng.binomial(n=1, p=h_prob, size=h_prob.shape)
        return (h_prob, h_sample)

    def sample_v_given_h(self, h):
        v_prob = sigmoid(self.params['v_bias'] + np.dot(h, self.params['W'].T))
        v_sample = self.rng.binomial(n=1, p=v_prob, size=v_prob.shape)
        return (v_prob, v_sample)


    def reconstruct_error(self, v):
        h_prob, h_sample = self.sample_h_given_v(v)
        v_prob, v_sample = self.sample_v_given_h(h_sample)

        cross_entropy =  - np.mean(np.sum(v * np.log(v_prob) + \
            (1 - v) * np.log(1 - v_prob), axis=1))

        return cross_entropy


    def sample_imgs(self, n):
        v_prob = np.random.uniform(low=0,high=1,size=(n, 28*28))
        v_sample = self.rng.binomial(n=1, p=v_prob, size=v_prob.shape)

        h_prob, h_sample = self.sample_h_given_v(v_sample)
        #negative phase
        for i in range(100):
            if i%20 == 0:
                print(i)
            v_prob, v_sample = self.sample_v_given_h(h_sample)
            h_prob, h_sample = self.sample_h_given_v(v_sample)
        return v_prob









