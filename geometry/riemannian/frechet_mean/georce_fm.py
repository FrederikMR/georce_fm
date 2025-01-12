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
from geometry.line_search import Backtracking

#%% Gradient Descent Estimation of Geodesics

class GEORCE_FM(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
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
    
    def init_curve(self, 
                   z_obs:Array, 
                   z_mu:Array,
                   )->Array:
        
        return vmap(self.init_fun, in_axes=(0,None,None))(z_obs, z_mu, self.T)
    
    def energy(self, 
               zt:Array,
               *args,
               )->Array:

        zt = zt.reshape(self.N, -1, self.dim)
        
        path_energy = vmap(self.path_energy, in_axes=(0,0,0))(self.z_obs, zt, self.G0)
        
        return jnp.sum(self.wi*path_energy)
    
    def path_energy(self, 
                    z0:Array,
                    zt:Array,
                    G0:Array,
                    )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt[:-1])
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt, term2)
        
        return val1+jnp.sum(val2)
    
    def Denergy(self,
                zt:Array,
                *args,
                )->Array:

        return grad(self.energy, argnums=0)(zt,*args)/self.N
        
    def Denergy_frechet(self,
                        zt:Array,
                        ut:Array,
                        z_mu:Array,
                        Gt:Array,
                        gt:Array,
                        )->Array:

        Gt = jnp.concatenate((self.G0.reshape(self.N, -1, self.dim, self.dim), 
                              Gt,
                              ),
                             axis=1)
        
        dcurve = jnp.mean(gt+2.*(jnp.einsum('...ij,...j->...i', Gt[:,:-1], ut[:,:-1])-\
                            jnp.einsum('...ij,...j->...i', Gt[:,1:], ut[:,1:])), axis=0)
        dmu = 2.*jnp.mean(jnp.einsum('...ij,...i->...j', Gt[:,-1], ut[:,-1]), axis=0)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(vmap(self.M.G))(zt)
        
        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut)), Gt
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        return lax.stop_gradient(grad(self.inner_product, has_aux=True)(zt,ut))
    
    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     gt_inv:Array,
                     ginv_sum_inv:Array,
                     )->Array:
        
        diff = jnp.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = jnp.sum(jnp.einsum('...ij,...j->...i', gt_inv[:,:-1], g_cumsum), axis=1)+2.0*diff

        muT = -jnp.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mut = jnp.concatenate((muT+g_cumsum, muT), axis=1)
        
        return mut
    
    def frechet_update(self,
                       g_cumsum:Array,
                       gt_inv:Array,
                       ginv_sum_inv:Array,
                       )->Array:
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', gt_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        lhs = jnp.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = jnp.linalg.solve(jnp.sum(lhs, axis=0), 
                              jnp.sum(rhs, axis=0),
                              )
        
        return mu
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  z_mu:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:

        return self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(alpha*ut_hat+(1-alpha)*ut, axis=1)

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, z_mu, gt, gt_inv, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, z_mu, gt, gt_inv, grad_norm, idx = carry
        
        g_cumsum = jnp.cumsum(gt[:,::-1], axis=1)[:,::-1]
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gt_inv, ginv_sum_inv)
        mut = self.curve_update(z_mu_hat, g_cumsum, gt_inv, ginv_sum_inv)

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ut[:,:-1], axis=1)
        
        gt, Gt = self.gt(zt, ut[:,1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim),  jnp.linalg.inv(Gt)),
                                 axis=1)
        
        grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu, Gt, gt))
        
        return (zt, ut, z_mu, gt, gt_inv, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut, z_mu = carry
        
        gt, Gt = self.gt(zt, ut[:,1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim),  jnp.linalg.inv(Gt)),
                                 axis=1)
        
        g_cumsum = jnp.cumsum(gt[:,::-1], axis=1)[:,::-1]
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gt_inv, ginv_sum_inv)
        mut = self.curve_update(z_mu_hat, g_cumsum, gt_inv, ginv_sum_inv)

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ut[:,:-1], axis=1)
        
        return ((zt, ut, z_mu),)*2

    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Denergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.z_obs = z_obs
        self.N, self.dim = self.z_obs.shape
        
        self.G0 = lax.stop_gradient(vmap(self.M.G)(self.z_obs))
        self.Ginv0 = lax.stop_gradient(jnp.linalg.inv(self.G0))
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zt = self.init_curve(self.z_obs, z_mu_init)
        ut = jnp.ones((self.N, self.T, self.dim), dtype=z_obs.dtype)*(z_mu_init-self.z_obs.reshape(-1,1,self.dim))/self.T
        
        if step == "while":
            gt, Gt = self.gt(zt, ut[:,1:])
            gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim), 
                                      jnp.linalg.inv(Gt),
                                      ),
                                     axis=1)
            grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu_init, Gt, gt))
            
            zt, _, z_mu, _, _, grad_norm, idx = lax.while_loop(self.cond_fun, 
                                                               self.while_step, 
                                                               init_val=(zt, ut, z_mu_init, gt, gt_inv, grad_norm, 0),
                                                               )
            
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut, z_mu_init),
                              xs=jnp.ones(self.max_iter),
                              )
            
            z_mu = val[2]
            zt = val[0]
            grad_norm = None
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return z_mu, zt, grad_norm, idx