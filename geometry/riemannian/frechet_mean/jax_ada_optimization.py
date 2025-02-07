#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.riemannian.manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class JAXAdaOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.seed = seed
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def random_batch(self,
                     )->Array:

        if self.batch_size > 0:
            
            self.key, subkey = jrandom.split(self.key)
            
            batch_idx = jrandom.choice(subkey, 
                                       a=self.batch,
                                       shape=(self.batch_size,), 
                                       replace=False,
                                      )
            
            return batch_idx
            
        else:
            return []
    
    def init_curve(self, 
                   z_obs:Array, 
                   z_mu:Array,
                   )->Array:
        
        return vmap(self.init_fun, in_axes=(0,None,None))(z_obs, z_mu, self.T)
    
    def energy(self, 
               z:Array,
               batch_idx:Array,
               )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu)

            return (energy,)*2
        
        zt = z[:-1].reshape(self.N,-1,self.dim)
        z_mu = z[-1]
        
        if self.batch_size > 0:
            zt = zt.at[batch_idx].set(lax.stop_gradient(zt[batch_idx]))
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zt, self.z_obs, self.wi, self.G0),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zt:Array,
                    G0:Array,
                    z_mu:Array,
                    )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = z_mu-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def obj_fun(self, 
                z:Array,
                batch_idx:Array,
                )->Array:
        
        energy = self.energy(z, batch_idx)
        
        return energy
    
    def Dobj(self,
             z:Array,
             batch_idx:Array,
             )->Array:
        
        return grad(self.obj_fun)(z, batch_idx)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        z, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        z, grad, opt_state, idx = carry
        
        batch_idx = self.batch_idx[idx]
        opt_state = self.opt_update(idx, grad, opt_state)
        z = self.get_params(opt_state)

        grad = self.Dobj(z, batch_idx)
        
        return (z, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 batch_idx,
                 )->Array:
        
        z, opt_state = carry

        grad = self.Dobj(z, batch_idx)
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z = self.get_params(opt_state)
        
        return ((z, opt_state),)*2
    
    def __call__(self, 
                 z_obs:Array,
                 batch_size:int=None,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.key = jrandom.key(self.seed)
        
        self.z_obs = z_obs
        self.G0 = vmap(self.M.G)(self.z_obs)
        self.dtype = self.z_obs.dtype
        self.N, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
            
        if batch_size is None:
            self.batch_size = 0
        else:
            self.batch_size = self.N-batch_size
            
        self.batch = jnp.arange(0,self.N, 1)
        batch_idx = self.random_batch()
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(z_obs, axis=0)
            
        zt = self.init_curve(self.z_obs, z_mu_init).reshape(-1,self.dim)
        z = jnp.vstack((zt, z_mu_init))
        
        opt_state = self.opt_init(z)
        
        if self.batch_size > 0:
            self.batch_idx = jnp.stack([self.random_batch() for i in jnp.ones(self.max_iter)])
        else:
            self.batch_idx = jnp.ones(self.max_iter)
        
        if step == "while":
            grad = self.Dobj(z, batch_idx)
        
            z, grad, _, idx = lax.while_loop(self.cond_fun,
                                             self.while_step,
                                             init_val=(z, grad, opt_state, 0),
                                             )
            
            zt = z[:-1].reshape(self.N, -1, self.dim)
            z_mu = z[-1]
            
            zt = zt[:,::-1]
            
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(z, opt_state),
                              xs = self.batch_idx,
                              )
            
            z = val[0]
            zt = z[:,:-1].reshape(self.max_iter, self.N, -1, self.dim)
            z_mu = z[:,-1]
            
            grad = vmap(self.Dobj)(z)
            idx = self.max_iter
            
            zt = zt[:,:,::-1]
            
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return z_mu, zt, grad, idx