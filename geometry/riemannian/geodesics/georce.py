#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.riemannian.manifolds import RiemannianManifold
from geometry.line_search import Backtracking, Bisection

#%% Gradient Descent Estimation of Geodesics

class GEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 max_iter:int=1000,
                 line_search_method:str="soft",
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        
        if line_search_method in ['soft', 'exact']:
            self.line_search_method = line_search_method
        else:
            raise ValueError(f"Invalid value for line search method, {line_search_method}")

        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def energy(self, 
               zt:Array,
               *args,
               )->Array:
        
        term1 = zt[0]-self.z0
        val1 = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(lambda z: self.M.G(z))(zt)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        return grad(self.inner_product)(zt,ut)
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return (self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1-alpha)*ut[:-1], axis=0),)
    
    def step(self,
             carry:Tuple[Array,Array],
             idx:int,
             )->Array:
        
        zt, ut = carry
        
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, vmap(lambda z: self.M.Ginv(z))(zt)))
        
        mut = self.unconstrained_opt(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        return ((zt, ut),)*2
    
    def unconstrained_opt(self, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        dtype = z0.dtype
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_xt,
                                            grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_xt,
                                         **self.line_search_params,
                                         )
        
        self.z0 = z0
        self.G0 = self.M.G(z0)
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        self.zT = zT
        self.diff = zT-z0
        
        ut = jnp.ones((self.T, self.dim), dtype=dtype)*self.diff/self.T

        _, val = lax.scan(self.step,
                          init=(zt, ut),
                          xs=jnp.ones(self.max_iter),
                          )
        
        zt = val[0][-1]
            
        return zt

        