import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class rSVD():
    def __init__(self,r, q, p, perm, dist):
        self.r = r # target rank
        self.q = q # power iteration 
        self.p = p # oversampling, small number (5 or 10)
        self.perm = perm
        self.distribution = dist        
        
    def call(self, X):
        ny = X.shape[-1]
        # Random projection
        if len(self.perm) == 3:
            P = tf.squeeze(self.distribution((X.shape[0], ny, self.r + self.p)))
        else:
            P = tf.squeeze(self.distribution((ny, self.r + self.p)))
        Z = tf.matmul(X, P) # Shrink column space
        for k in range(self.q):
            Z = tf.matmul(X, tf.matmul(tf.transpose(X, perm=self.perm), Z))
    
        Q, R = tf.linalg.qr(Z, full_matrices=False) # Q- orthonormal basis for Z (and also X)
        # Project X into Q
        Y = tf.matmul(tf.transpose(Q, perm=self.perm), X)
        
        n = Y.shape[0]
        m = Y.shape[1]
        S, UY, VT = tf.linalg.svd(Y, full_matrices=False)
        U = tf.matmul(Q, UY)
        # S and V same for X
    
        return S, U, VT

class SVD():
    def call(self,X):
        n = X.shape[0]
        m = X.shape[1]        
        return tf.linalg.svd(X)
    
class PCA_Abstract(tf.keras.Model):
    def __init__(self, n_components, data_dim, perm, incremental, svdkwargs):
        super().__init__()
        self.data_dim = data_dim
        self.n_components = n_components
        self.features = tf.reduce_prod(data_dim)
        self.perm = perm
        self.incremental = incremental
        if svdkwargs['use_rSVD']:
            self.svd = rSVD(r=n_components,p=svdkwargs['p'], q=svdkwargs['q'], perm=perm, dist = svdkwargs['dist']).call
        else:
            self.svd = SVD().call
            
    def _encode(self, x, n_comp, shape):        
        v = self.components
        if n_comp is not None:
            v = v[...,:n_comp]        
        x_len = len(x.shape)
        x = tf.reshape(x, shape)
        x -= self.mean
        z = tf.matmul(x, v)

        return z
        
    def _decode(self, z, n_comp):
        v = self.components
        if n_comp is not None:
            v = v[...,:n_comp]         
        X = tf.matmul(z, tf.transpose(v, perm=self.perm))
        X += self.mean
        return X
    
    def encode(self, data, n_comp=None):
        raise NotImplemented()
    
    def decode(self, X, shape, n_comp=None):
        raise NotImplemented()
    
    def reconstruct(self, X, n_comp=None):
        return self.decode(self.encode(X, n_comp), X.shape, n_comp)
    
    def train(self, X):
        if self.incremental:
            return self.incremental_fit(X)
        return self.fit(X)
    
    def fit(self, X):
        raise NotImplemented()
        
    def _fit(self, X, shape, axis):
        X = tf.reshape(X, shape)

        # Reduce mean
        mean = tf.reduce_mean(X, axis=axis, keepdims=True)
        var = tf.math.reduce_variance(X, axis=axis, keepdims=True)
        X -= mean

        # Svd        
        singular_values, u, v = self.svd(X)

        # Sign correction
        v = -v
        abs_v = tf.abs(v)
        m = tf.equal(abs_v, tf.reduce_max(abs_v, axis=-2, keepdims=True))
        m = tf.cast(m, v.dtype)
        signs = tf.sign(tf.reduce_sum(v * m, axis=-2, keepdims=True))
        v *= signs

        explained_variance = (singular_values**2) / (X.shape[0] - 1)
        total_var = tf.reduce_sum(explained_variance)
        explained_variance_ratio = explained_variance / total_var
        explained_variance = explained_variance[...,:self.n_components]
        explained_variance_ratio = explained_variance_ratio[...,:self.n_components]

        components = v[...,:self.n_components]
        singular_values = singular_values[...,:self.n_components]
        return mean, components, singular_values, var, explained_variance, explained_variance_ratio
    
    def incremental_fit(self, ds):
        n_samples_seen = 0
        for data in tqdm(ds):
            n_samples_seen = self.incremental_step(data, n_samples_seen)
    
    def incremental_step(self, data, n_samples_seen):
        raise NotImplemented()
        
    def _incremental_step(self, data, mean, var, components, singular_values, n_samples_seen, axis, shape):        
        batch_size = data.shape[axis]            
        assert self.n_components <= batch_size, f'{self.n_components}, {batch_size}'
        n_samples = batch_size

        X = data
        X = tf.reshape(X, shape)

        data_mean = tf.reduce_mean(X, axis=axis, keepdims=True)
        data_var = tf.math.reduce_variance(X, axis=axis, keepdims=True)

        updated_mean = mean * n_samples_seen
        updated_mean += (data_mean)*n_samples
        updated_mean /= (n_samples_seen + n_samples)

        updated_var = (n_samples_seen/(n_samples_seen+n_samples))*(var + mean**2) + (n_samples/(n_samples_seen+n_samples))*(data_var + data_mean**2) - updated_mean**2

        if n_samples_seen == 0:
            X -= updated_mean
        else:
            mean_correction = tf.sqrt((n_samples_seen * n_samples) / (n_samples_seen + n_samples)) * (mean - data_mean)
            SV = singular_values[...,None] * tf.transpose(components, perm=self.perm)            
            X_tilde = X - data_mean
            X = tf.concat([SV,
                           X_tilde,
                           mean_correction], axis=axis)
        n_samples_seen += n_samples
        singular_values, _, v = self.svd(X)

        # Sign correction
        v = -v
        abs_v = tf.abs(v)
        m = tf.equal(abs_v, tf.reduce_max(abs_v, axis=-2, keepdims=True))
        m = tf.cast(m, v.dtype)
        signs = tf.sign(tf.reduce_sum(v * m, axis=-2, keepdims=True))
        v *= signs

        explained_variance = singular_values**2 / (n_samples_seen - 1)
        explained_variance_ratio = singular_values**2 / np.sum(updated_var * n_samples_seen)

        explained_variance = explained_variance[...,:self.n_components]
        explained_variance_ratio = explained_variance_ratio[...,:self.n_components]
        components = v[...,:self.n_components]        
        singular_values = singular_values[...,:self.n_components]
        mean = updated_mean
        var = updated_var
        return n_samples_seen, mean, components, singular_values, var, explained_variance, explained_variance_ratio

class PCA(PCA_Abstract):
    def __init__(self, n_components, data_dim, incremental, **rSvdkwargs):
        super().__init__(n_components, data_dim, [1,0], incremental, rSvdkwargs)
        
        self.mean = self.add_weight(name='mean', shape=(1, self.features), initializer=tf.zeros_initializer())
        self.components = self.add_weight(name='components', shape=(self.features,self.n_components), initializer=tf.zeros_initializer())
        self.singular_values = self.add_weight(name='singular_values', shape=(self.n_components, ), initializer=tf.zeros_initializer())
        self.var = self.add_weight(name='variance', shape=(1, self.features, ), initializer=tf.zeros_initializer())
        self.explained_variance = self.add_weight(name='explained variance', shape=(self.n_components, ), initializer=tf.zeros_initializer())
        self.explained_variance_ratio = self.add_weight(name='explained variance ratio', shape=(self.n_components, ), initializer=tf.zeros_initializer())
    
    def encode(self, data, n_comp=None):
        return super()._encode(data, n_comp, (data.shape[0], -1))
    
    def decode(self, Z, shape, n_comp=None):
        X = super()._decode(Z, n_comp)
        return tf.reshape(X, shape)            
    
    def fit(self, X):
        mean, components,singular_values, var, explained_variance, explained_variance_ratio = self._fit(X, 
                                                                                                        (X.shape[0], -1),
                                                                                                        axis=0)
        self.set_weights([mean, components,singular_values, var, explained_variance, explained_variance_ratio]) 
        
        
    def incremental_step(self, data, n_samples_seen):
        n_samples_seen, m, v, sv, var, ev, evr = super()._incremental_step(data,
                                                                           self.mean, 
                                                                           self.var, 
                                                                           self.components, 
                                                                           self.singular_values,
                                                                           n_samples_seen,
                                                                           axis=0,
                                                                           shape = (data.shape[0], -1))
        self.set_weights([m, v, sv, var, ev, evr])
        return n_samples_seen    

class multichannel_PCA(PCA_Abstract):
    def __init__(self, n_components, data_dim, incremental, **rSvdkwargs):
        assert len(data_dim) == 3
        super().__init__(n_components, data_dim, [0,2,1], incremental, rSvdkwargs)
        self.channels = 3
        self.mean = self.add_weight(name='mean', shape=(self.channels, 1, self.features), initializer=tf.zeros_initializer())
        self.components = self.add_weight(name='components', shape=(self.channels, self.features,self.n_components), initializer=tf.zeros_initializer())
        self.singular_values = self.add_weight(name='singular_values', shape=(self.channels, self.n_components), initializer=tf.zeros_initializer())
        self.var = self.add_weight(name='variance', shape=(self.channels, 1, self.features), initializer=tf.zeros_initializer())
        self.explained_variance = self.add_weight(name='explained variance', shape=(self.channels, self.n_components), initializer=tf.zeros_initializer())
        self.explained_variance_ratio = self.add_weight(name='explained variance ratio', shape=(self.channels, self.n_components), initializer=tf.zeros_initializer())
    
    def encode(self, data, n_comp=None):
        data = tf.einsum("ndhwc->cndhw", data)
        return super()._encode(data, n_comp, (data.shape[0], data.shape[1], -1))
        
    def decode(self, X, shape, n_comp=None):
        X = super()._decode(X, n_comp)
        X = tf.einsum('cdf->dfc', X)
        return tf.reshape(X, shape)
    
    def fit(self, data):
        data = tf.einsum("ndhwc->cndhw", data)
        M, V,SV, VAR, EV, EVR = self._fit(data, (data.shape[0], data.shape[1], -1), axis=1)        
        self.set_weights([M, V,SV, VAR, EV, EVR])
        
        
    def incremental_step(self, data, n_samples_seen):
        data = tf.einsum("ndhwc->cndhw", data)
        
        n_samples_seen, M, V,SV, VAR, EV, EVR = super()._train_step(data,
                                                                    self.mean,
                                                                    self.var,
                                                                    self.components,
                                                                    self.singular_values,
                                                                    n_samples_seen,
                                                                    axis=1,
                                                                    shape=(data.shape[0], data.shape[1], -1))        
        
        self.set_weights([M, V,SV, VAR, EV, EVR])
        return n_samples_seen
