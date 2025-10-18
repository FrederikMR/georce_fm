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

class KarcherFlow(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 order:float=1.0,
                 )->None:
        
        self.M = M
        self.max_iter = max_iter
        self.tol = tol
        self.order = order
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def H_fun(self, 
              x:Array,
              y:Array,
              )->Array:
        
        v = self.M.Log(x,y)
        dist = self.M.dist(x,y)
        
        return lax.cond(dist < 1e-6,
                        lambda *_: jnp.zeros_like(v, dtype=v.dtype),
                        lambda *_: v / dist,
                        )
    
    
    
    def Dobj(self,
             mu:Array,
             )->Array:
        
        v = jnp.sum(vmap(self.M.Log, in_axes=(None,0))(mu, self.x_obs), axis=0)
        
        return v
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        mu, grad_val, idx = carry

        norm_grad = jnp.linalg.norm(grad_val.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple,
                   )->Array:
        
        mu, grad_val, idx = carry

        grad_val = self.Dobj(mu)
        
        H = jnp.mean(vmap(self.H_fun, in_axes=(None,0))(mu, self.x_obs), axis=0)
        
        v = H/jnp.linalg.norm(H)
        
        mu = self.M.Exp(mu, v/((idx+1)**self.order))

        return (mu, grad_val, idx+1)
    
    def for_step(self,
                 z:Array,
                 idx:int,
                 )->Array:
        
        grad_val = self.Dobj(z)
                
        H = jnp.mean(vmap(self.H_fun, in_axes=(None,0))(mu, self.x_obs), axis=0)
        
        v = H/jnp.linalg.norm(H)
        
        mu = self.M.Exp(mu, v/((idx+1)**self.order))
        
        return (mu,)*2
    
    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.z_obs = z_obs
        self.x_obs = vmap(self.M.f)(z_obs)

        self.N = self.z_obs.shape[0]
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)
            
        x_mu_init = self.M.f(z_mu_init)
        
        if step == "while":
            grad_val = self.Dobj(x_mu_init)
        
            x_mu, grad, idx = lax.while_loop(self.cond_fun,
                                             self.while_step,
                                             init_val=(x_mu_init, grad_val, 0),
                                             )
            
            z_mu = self.M.invf(x_mu)
            
        elif step == "for":
            _, x_mu = lax.scan(self.for_step,
                              init=x_mu,
                              xs = jnp.ones(self.max_iter),
                              )
            
            z_mu = vmap(self.M.invf)(x_mu)

            grad_val = None
            idx = self.max_iter
            
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return z_mu, grad_val, idx