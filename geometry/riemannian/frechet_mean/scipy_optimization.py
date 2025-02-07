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

class ScipyOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.M = M
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def init_curve(self, 
                   z_obs:Array, 
                   z_mu:Array,
                   )->Array:
        
        return vmap(self.init_fun, in_axes=(0,None,None))(z_obs, z_mu, self.T)
    
    def energy(self, 
               z:Array,
               )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu)

            return (energy,)*2
        
        zt = z[:-1].reshape(self.N,-1,self.dim)
        z_mu = z[-1]
        
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
                )->Array:
        
        z = z.reshape(-1,self.dim)
        
        energy = self.energy(z)
        
        return jnp.sum(energy)
    
    def Dobj(self,
             z:Array,
             )->Array:
        
        return grad(self.obj_fun)(z)
    
    def HessObj(self,
                z:Array,
                )->Array:
        
        return hessian(self.obj_fun)(z)
    
    def HessPEnergy(self,
                   z:Array,
                   p:Array,
                   )->Array:
        
        hess = self.HessObj(z)
        
        return jnp.dot(hess, p)
    
    def callback(self,
                 z:Array
                 )->Array:
        
        self.save_zt.append(z)
        
        return
    
    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.z_obs = z_obs
        
        self.G0 = vmap(self.M.G)(self.z_obs)
        self.dtype = self.z_obs.dtype
        self.N, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)
            
        zt = self.init_curve(self.z_obs, z_mu_init).reshape(-1,self.dim)
        z = jnp.vstack((zt, z_mu_init)).reshape(-1)
        
        if step == "while":
            res = minimize(fun = self.obj_fun, 
                           x0=z, 
                           method=self.method, 
                           jac=self.Dobj,
                           hess=self.HessObj,
                           hessp=self.HessPEnergy,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
            
            z = jnp.array(res.x).reshape(-1,self.dim)
            
            zt = z[:-1].reshape(self.N, -1, self.dim)
            z_mu = z[-1]

            grad =  jnp.array(res.jac)
            idx = res.nit
            
            zt = zt[:,::-1]
            
        elif step == "for":
            res = minimize(fun = self.obj_fun,
                           x0=z,
                           method=self.method,
                           jac=self.Dobj,
                           hess=self.HessObj,
                           hessp=self.HessPEnergy,
                           callback=self.callback,
                           tol=self.tol,
                           options={'maxiter': self.max_iter}
                           )
            
            zt = jnp.stack([z[:-1].reshape(self.N,-1,self.dim) for z in self.save_zt])
            zt = jnp.stack([z[-1].reshape(self.dim) for z in self.save_zt])
            zt = jnp.stack([z for z in self.save_zt])

            grad = vmap(self.Dobj)(z)
            idx = self.max_iter
            
            zt = zt[:,:,::-1]
            

        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return z_mu, zt, grad, idx
    
    
    