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

class ScipyOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 sub_iter:int=10,
                 rho:float=0.5,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                              1.0,
                                                              T,
                                                              endpoint=False,
                                                              dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            init_fun = init_fun
            
        self.M = M
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.Geodesic = GEORCE(M=M,
                               init_fun=init_fun,
                               T=T,
                               max_iter=sub_iter,
                               line_search_method="soft",
                               line_search_params = {'rho':rho},
                               )
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
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
        
        return jnp.sum(energy)
    
    def Dobj(self,
             z_mu:Array,
             )->Array:
        
        return grad(self.obj_fun)(z_mu)
    
    def HessObj(self,
                z_mu:Array,
                )->Array:
        
        return hessian(lambda z: self.obj_fun(z)[0])(zt)
    
    def HessPEnergy(self,
                   z_mu:Array,
                   p:Array,
                   )->Array:
        
        hess = self.HessObj(z_mu)
        
        return jnp.dot(hess, p)
    
    def callback(self,
                 z_mu:Array
                 )->Array:
        
        self.save_zt.append(z_mu)
        
        return
    
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
        
        #if self.method == "BFGS":
        #    min_fun = jminimize
        #else:
        min_fun = minimize
        
        if step == "while":
            res = min_fun(fun = self.obj_fun, 
                          x0=z_mu_init, 
                          method=self.method, 
                          jac=self.Dobj,
                          hess=self.HessObj,
                          hessp=self.HessPEnergy,
                          tol=self.tol,
                          options={'maxiter': self.max_iter}
                          )
        
            z_mu = jnp.array(res.x)
            grad =  jnp.array(res.jac)
            idx = res.nit
        elif step == "for":
            res = min_fun(fun = self.obj_fun,
                          x0=z_mu_init,
                          method=self.method,
                          jac=self.Dobj,
                          hess=self.HessObj,
                          hessp=self.HessPEnergy,
                          callback=self.callback,
                          tol=self.tol,
                          options={'maxiter': self.max_iter}
                          )
            
            z_mu = jnp.stack([z_mu.reshape(-1,self.dim) for zt in self.save_zt])
            
            grad, vals = vmap(self.Dobj)(z_mu)
            idx = self.max_iter

        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        dist = None
        zt = None
        
        return z_mu, zt, dist, grad, idx
    
    
    