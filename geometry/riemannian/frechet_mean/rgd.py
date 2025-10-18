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
from geometry.line_search import Backtracking

#%% Gradient Descent Estimation of Geodesics

class RGD(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 lr_rate:float=0.01,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.M = M
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        self.tol = tol
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
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
                
        mu = self.M.Exp(mu, self.lr_rate*grad_val)

        return (mu, grad_val, idx+1)
    
    def for_step(self,
                 z:Array,
                 idx:int,
                 )->Array:
        
        grad_val = self.Dobj(z)
                
        mu = self.M.Exp(mu, self.lr_rate*grad_val)
        
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
    
#%% Gradient Descent Estimation of Geodesics

class RGD_LineSearch(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_params = line_search_params
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def length(self,
               mu:Array,
               *args,
               )->Array:
        
        dist = vmap(self.M.dist, in_axes=(None,0))(mu,self.x_obs)
        
        return jnp.sum(dist**2)
    
    def Dobj(self,
             mu:Array,
             *args,
             )->Array:
        
        v = jnp.sum(vmap(self.M.Log, in_axes=(None,0))(mu, self.x_obs), axis=0)
        
        return v
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        mu, grad_val, idx = carry

        norm_grad = jnp.linalg.norm(grad_val.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def update(self,
               mu:Array,
               alpha:float,
               v:Array,
               )->Array:
        
        return (self.M.Exp(mu, alpha*v),)
    
    def while_step(self,
                   carry:Tuple,
                   )->Array:
        
        mu, grad_val, idx = carry

        grad_val = self.Dobj(mu)
        tau = self.line_search((mu,), grad_val)
        mu = self.M.Exp(mu, tau*grad_val)

        return (mu, grad_val, idx+1)
    
    def for_step(self,
                 z:Array,
                 idx:int,
                 )->Array:
        
        grad_val = self.Dobj(z)
                
        tau = self.line_search((mu,), grad_val)
        mu = self.M.Exp(mu, tau*grad_val)
        
        return (mu,)*2
    
    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.length,
                                        update_fun=self.update,
                                        grad_fun = lambda z,*args: self.Dobj(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        criterion="naive",
                                        )
        
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