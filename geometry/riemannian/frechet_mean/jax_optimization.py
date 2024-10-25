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
from geometry.riemannian.geodesics import GEORCE

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 sub_iter:int=10,
                 tol:float=1e-4,
                 rho:float=0.5,
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
            init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                              1.0,
                                                              T,
                                                              endpoint=False,
                                                              dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            init_fun = init_fun
            
        self.Geodesic = GEORCE(M=M,
                               init_fun=init_fun,
                               T=T,
                               max_iter=sub_iter,
                               line_search_method="soft",
                               line_search_params = {'rho':rho},
                               )
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def energy_frechet(self, 
                       zt:Array,
                       z_mu:Array,
                       )->Array:

        zt = zt.reshape(self.N, -1, self.dim)
        
        path_energy = vmap(self.path_energy_frechet, in_axes=(0,0,None,0))(self.z_obs, zt, z_mu, self.G0)
        
        return jnp.sum(path_energy)
    
    def length_frechet(self, 
                       zt:Array,
                       z_mu:Array,
                       )->Array:

        zt = zt.reshape(self.N, -1, self.dim)
        
        path_length = vmap(self.path_length_frechet, in_axes=(0,0,None,0))(self.z_obs, zt, z_mu, self.G0)
        
        return jnp.sum(path_length**2)
    
    def path_length_frechet(self, 
                            z0:Array,
                            zt:Array,
                            mu:Array,
                            G0:Array,
                            )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.sqrt(jnp.einsum('i,ij,j->', term1, G0, term1))
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.sqrt(jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2))
        
        term3 = mu-zt[-1]
        val3 = jnp.sqrt(jnp.einsum('i,ij,j->', term3, Gt[-1], term3))
        
        return (val1+jnp.sum(val2)+val3)**2
    
    def path_energy_frechet(self, 
                            z0:Array,
                            zt:Array,
                            mu:Array,
                            G0:Array,
                            )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = mu-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return (val1+jnp.sum(val2)+val3)**2
    
    def obj_fun(self, 
                z_mu:Array, 
                )->Array:
        
        zt_curves = vmap(self.Geodesic, in_axes=(0,None))(self.z_obs, z_mu)
        
        energy = self.energy_frechet(zt_curves, z_mu)
        dist = self.length_frechet(zt_curves, z_mu)
        
        return jnp.sum(energy), (zt_curves, dist)
    
    def Dobj(self,
             z_mu:Array,
             )->Array:
        
        return grad(self.obj_fun, has_aux=True)(z_mu)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        z_mu, zt, dist, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        z_mu, zt, dist, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z_mu = self.get_params(opt_state)

        grad, vals = self.Dobj(z_mu)
        zt, dist = vals
        
        return (z_mu, zt, dist, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        z_mu, zt, dist, opt_state = carry
        
        grad, vals = self.Dobj(z_mu)
        zt, dist = vals
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z_mu = self.get_params(opt_state)
        
        return ((z_mu, zt, dist, opt_state),)*2
    
    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.z_obs = z_obs.astype('float64')
        
        self.G0 = vmap(self.M.G)(self.z_obs)
        self.dtype = self.z_obs.dtype
        self.N, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = self.z_obs[0]#jnp.mean(self.z_obs, axis=0)
        
        opt_state = self.opt_init(z_mu_init)
        
        if step == "while":
            grad, vals = self.Dobj(z_mu_init)
            zt, dist = vals
        
            z_mu, zt, dist, grad, _, idx = lax.while_loop(self.cond_fun, 
                                                          self.while_step,
                                                          init_val=(z_mu_init, zt, dist, grad, opt_state, 0)
                                                          )
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(z_mu_init, 
                                    jnp.zeros(self.N, self.T-1, self.dim),
                                    0.0,
                                    opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            z_mu, zt, dist = val[0], val[1], val[2]
            
            grad, vals = vmap(self.Dobj)(z_mu)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return z_mu, zt, dist, grad, idx