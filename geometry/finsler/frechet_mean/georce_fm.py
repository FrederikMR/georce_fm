#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.finsler.manifolds import FinslerManifold
from geometry.line_search import Backtracking

#%% Frechet Mean using GEORCE_FM

class GEORCE_FM(ABC):
    def __init__(self,
                 M:FinslerManifold,
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
                   z0:Array, 
                   zT:Array,
                   )->Array:
        
        zt = self.init_fun(z0, zT, self.T)
        total = jnp.vstack((z0, zt, zT))
        ut = total[1:]-total[:-1]
        
        return zt, ut
    
    def energy(self, 
               zt:Array,
               z_mu:Array,
               *args,
               )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w = y
            
            energy += w*self.path_energy(z_obs, z, z_mu)

            return (energy,)*2
        
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
                        ht:Array,
                        )->Array:
        
        dcurve = jnp.mean(gt + 2.*(jnp.einsum('...ij,...j->...i', Gt[:,:-1], ut[:,:-1])-\
                                   jnp.einsum('...ij,...j->...i', Gt[:,1:], ut[:,1:])) \
                          + ht[:,:-1] - ht[:,1:], axis=0)
        dmu = 2.*jnp.mean(jnp.einsum('...ij,...i->...j', Gt[:,-1], ut[:,-1])+ht[:,-1], axis=0)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(self.M.G)(zt,-ut)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut))
    
    def inner_product_h(self,
                        zt:Array,
                        u0:Array,
                        ut:Array,
                        )->Array:
        
        Gt = vmap(self.M.G)(zt,-ut)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', u0, Gt, u0)), Gt
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        def step_gt(g:Array,
                    y:Tuple,
                    )->Tuple:
            
            z,u = y
            
            g = lax.stop_gradient(grad(self.inner_product)(z, u))
            
            return (g,)*2
        
        _, gt = lax.scan(step_gt,
                         init=jnp.zeros((self.T-1, self.dim), dtype=zt.dtype),
                         xs=(zt,ut),
                         )

        return gt
    
    def ht(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        def step_ht(c:Tuple,
                    y:Tuple,
                    )->Tuple:
            
            z,u = y
            
            h, G = lax.stop_gradient(grad(self.inner_product_h, argnums=2, has_aux=True)(z, u, u))
            
            return ((h,G),)*2
        
        _, (ht, Gt) = lax.scan(step_ht,
                               init=(jnp.zeros((self.T, self.dim), dtype=zt.dtype),
                                     jnp.zeros((self.T, self.dim,self.dim), dtype=zt.dtype)),
                               xs=(zt,ut),
                               )
        
        return ht, Gt

    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     gt_inv:Array,
                     ginv_sum_inv:Array,
                     ht:Array,
                     )->Array:
        
        diff = jnp.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = jnp.sum(jnp.einsum('...ij,...j->...i', gt_inv, g_cumsum+ht), axis=1)+2.0*diff

        muT = -jnp.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mut = muT+g_cumsum+ht
        
        return mut
    
    def frechet_update(self,
                       g_cumsum:Array,
                       gt_inv:Array,
                       ginv_sum_inv:Array,
                       ht:Array,
                       )->Array:
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', gt_inv, g_cumsum+ht), axis=1),
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
                  z_mu_hat:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        x_new = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(alpha*ut_hat+(1-alpha)*ut, axis=1)
        z_mu_new = alpha*z_mu_hat+(1.-alpha)*z_mu

        return (x_new, z_mu_new)

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, z_mu, ht, gt, gt_inv, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, z_mu, ht, gt, gt_inv, grad_norm, idx = carry
        
        g_cumsum = jnp.concatenate((jnp.cumsum(gt[:,::-1], axis=1)[:,::-1], jnp.zeros((self.N, 1, self.dim))), axis=1)
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gt_inv, ginv_sum_inv, ht)
        mut = self.curve_update(z_mu_hat, g_cumsum, gt_inv, ginv_sum_inv, ht)

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ut[:,:-1], axis=1)
        
        gt = self.gt(zt, ut[:,1:])
        ht, Gt = self.ht(jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zt), axis=1), ut)
        gt_inv = jnp.linalg.inv(Gt)
        
        grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu, Gt, gt, ht))
        
        return (zt, ut, z_mu, ht, gt, gt_inv, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut, z_mu = carry
        
        gt = self.gt(zt, ut[:,1:])
        ht, Gt = self.ht(jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zt), axis=1), ut)
        gt_inv = jnp.linalg.inv(Gt)
        
        g_cumsum = jnp.concatenate((jnp.cumsum(gt[:,::-1], axis=1)[:,::-1], jnp.zeros((self.N, 1, self.dim))), axis=1)
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gt_inv, ginv_sum_inv, ht)
        mut = self.curve_update(z_mu_hat, g_cumsum, gt_inv, ginv_sum_inv, ht)

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu, z_mu_hat, ut_hat, ut)

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
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zt, ut = vmap(self.init_curve, in_axes=(0,None))(self.z_obs, z_mu_init)
        
        if step == "while":
            gt = self.gt(zt, ut[:,1:])
            ht, Gt = self.ht(jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zt), axis=1), ut)
            gt_inv = jnp.linalg.inv(Gt)
            grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu_init, Gt, gt, ht))
            
            zt, _, z_mu, _, _, _, grad_norm, idx = lax.while_loop(self.cond_fun, 
                                                                  self.while_step, 
                                                                  init_val=(zt, ut, z_mu_init, ht, gt, gt_inv, grad_norm, 0),
                                                                  )
            
            zt = zt[:,::-1]
            
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut, z_mu_init),
                              xs=jnp.ones(self.max_iter),
                              )
            
            z_mu = val[2]
            zt = val[0]
            grad_norm = None
            idx = self.max_iter
            
            zt = zt[:,:,::-1]
            
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return z_mu, zt, grad_norm, idx