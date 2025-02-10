#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.finsler.manifolds import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:FinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 parallel:bool=True,
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if parallel:
            self.energy = self.vmap_energy
        else:
            self.energy = self.loop_energy
        
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
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def init_curve(self, 
                   z_obs:Array, 
                   z_mu:Array,
                   )->Array:
        
        return vmap(self.init_fun, in_axes=(0,None,None))(z_obs, z_mu, self.T)
    
    def vmap_energy(self, 
                    z:Array,
                    )->Array:
        
        zt = z[:-1]
        z_mu = z[-1]
        zt = zt.reshape(self.N, -1, self.dim)

        energy = vmap(self.path_energy, in_axes=(0,0,None))(self.z_obs, zt, z_mu)

        return jnp.sum(self.wi*energy)
    
    def loop_energy(self, 
                    z:Array,
                    )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w = y
            
            energy += w*self.path_energy(z_obs, z, z_mu)

            return (energy,)*2
        
        zt = z[:-1]
        z_mu = z[-1]
        zt = zt.reshape(self.N, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zt, self.z_obs, self.wi),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zt:Array,
                    z_mu:Array,
                    )->Array:
        
        term1 = zt[0]-z0
        val1 = self.M.F(z0, -term1)**2
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda z,v: self.M.F(z,v)**2)(zt[:-1],-term2)
        
        term3 = z_mu-zt[-1]
        val3 = self.M.F(zt[-1],-term3)**2
        
        return val1+jnp.sum(val2)+val3
    
    def obj_fun(self, 
                z:Array,
                )->Array:
        
        energy = self.energy(z)
        
        return jnp.sum(energy)
    
    def Dobj(self,
             z:Array,
             )->Array:
        
        return grad(self.obj_fun)(z)
    
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
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z = self.get_params(opt_state)

        grad = self.Dobj(z)
        
        return (z, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        z, opt_state = carry
        
        grad = self.Dobj(z)
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z = self.get_params(opt_state)
        
        return ((z, opt_state),)*2
    
    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.z_obs = z_obs
        self.dtype = self.z_obs.dtype
        self.N, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)
            
        zt = self.init_curve(self.z_obs, z_mu_init).reshape(-1,self.dim)
        z = jnp.vstack((zt, z_mu_init))
        
        opt_state = self.opt_init(z)
        
        if step == "while":
            grad = self.Dobj(z)
        
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
                              xs = jnp.ones(self.max_iter),
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