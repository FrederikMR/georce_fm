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

#%% Adaptive GEORCE Frechet Estimation

class GEORCE_AdaFM(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=0.48,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 sub_iter:int=5,
                 conv_flag:float=1.0,
                 line_search_params:Dict = {'rho':0.5},
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.lam = lam
        self.T = T
        self.tol = tol
        self.conv_flag = conv_flag
        self.max_iter = max_iter
        self.sub_iter = sub_iter

        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.key = jrandom.key(seed)
        self.seed = seed
        
    def __str__(self)->str:
        
        return "Adaptive Fréchet Estimation usign GEORCE"
    
    def random_batch(self,
                     subkey,
                     )->Array:

        batch_idx = jrandom.choice(subkey, 
                                   a=self.batch,
                                   shape=(self.batch_size,), 
                                   replace=False,
                                  )
        
        return batch_idx
    
    def init_curve(self, 
                   z_obs:Array, 
                   z_mu:Array,
                   )->Array:
        
        return vmap(self.init_fun, in_axes=(0,None,None))(z_obs, z_mu, self.T)
    
    def update_default(self,
                       Wk1:Array,
                       Wk2:Array,
                       Vk1:Array,
                       Vk2:Array,
                       idx:int,
                       )->Tuple:
        
        alpha = self.lam/(1.-self.lam**(idx+1))
        
        Wk_hat = alpha*Wk1+(1.-alpha)*Wk2
        Vk_hat = alpha*Vk1+(1.-alpha)*Vk2
        
        return Wk_hat, Vk_hat
    
    def update_convergence(self,
                           Wk1:Array,
                           Wk2:Array,
                           Vk1:Array,
                           Vk2:Array,
                           idx:int,
                           )->Tuple:
        
        Wk_hat = Wk1*idx/(idx+1.)+Wk2/(idx+1.)
        Vk_hat = Vk1*idx/(idx+1.)+Vk2/(idx+1.)
        
        return Wk_hat, Vk_hat
    
    def update_mean(self,
                    Wk_hat,
                    Vk_hat,
                    )->Array:
        
        return jnp.linalg.solve(Wk_hat, Vk_hat)
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        z_mu, z_diff, Wk, Vk, idx = carry

        return (z_diff>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Array,
                   )->Array:
        
        z_mu, z_diff, Wk1, Vk1, idx = carry
        
        batch_idx = self.random_batch(self.subkeys[idx])
        
        z_batch, w_batch, G0_batch, Ginv0_batch = self.z_obs[batch_idx], self.wi[batch_idx], self.G0[batch_idx], self.Ginv0[batch_idx]
        Wk2, Vk2, grad_norm = self.georce_fm(z_batch, w_batch, G0_batch, Ginv0_batch)
        
        Wk_hat, Vk_hat = lax.cond(z_diff < self.conv_flag,
                                  lambda *args: self.update_convergence(*args),
                                  lambda *args: self.update_default(*args),
                                  Wk1,
                                  Wk2,
                                  Vk1,
                                  Vk2,
                                  idx,
                                  )
        
        z_mu_hat = self.update_mean(Wk_hat, Vk_hat)
        z_diff = jnp.linalg.norm(z_mu-z_mu_hat)
        
        return (z_mu_hat, z_diff, Wk_hat, Vk_hat, idx+1)
    
    def for_step(self,
                 carry:Array,
                 subkey,
                 )->Array:
        
        z_mu, z_diff, Wk1, Vk1, idx = carry
        batch_idx = self.random_batch(subkey)
        
        z_batch, w_batch, G0_batch, Ginv0_batch = self.z_obs[batch_idx], self.wi[batch_idx], self.G0[batch_idx], self.Ginv0[batch_idx]
        Wk2, Vk2, grad_norm = self.georce_fm(z_batch, w_batch, G0_batch, Ginv0_batch)
        
        Wk_hat, Vk_hat = lax.cond(z_diff < self.conv_flag,
                                  lambda *args: self.update_convergence(*args),
                                  lambda *args: self.update_default(*args),
                                  Wk1,
                                  Wk2,
                                  Vk1,
                                  Vk2,
                                  idx,
                                  )
        
        z_mu_hat = self.update_mean(Wk_hat, Vk_hat)
        z_diff = jnp.linalg.norm(z_mu-z_mu_hat)
        
        return ((z_mu_hat, z_diff, Wk_hat, Vk_hat, idx+1),)*2

    def __call__(self, 
                 z_obs:Array,
                 batch_size:int=None,
                 wi:Array=None,
                 step:str="while",
                 )->Array:
        
        self.z_obs = z_obs
        self.N, self.dim = self.z_obs.shape
        
        self.G0 = lax.stop_gradient(vmap(self.M.G)(self.z_obs))
        self.Ginv0 = lax.stop_gradient(jnp.linalg.inv(self.G0))
        
        if batch_size is None:
            self.batch_size = self.N
        else:
            self.batch_size = batch_size
            
        self.batch = jnp.arange(0,self.N, 1)
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        self.georce_fm = jit(GEORCE_FM_Step(self.M, 
                                            self.init_fun, 
                                            T=self.T, 
                                            iters=self.sub_iter,
                                            line_search_params=self.line_search_params,
                                            ))
        
        
        subkeys = jrandom.split(self.key, self.max_iter+1)
        self.subkeys = subkeys[1:]
        
        batch_idx = self.random_batch(subkeys[0])
        z_batch, w_batch, G0_batch, Ginv0_batch = self.z_obs[batch_idx], self.wi[batch_idx], self.G0[batch_idx], self.Ginv0[batch_idx]

        Wk, Vk, grad_norm = self.georce_fm(z_batch, w_batch, G0_batch, Ginv0_batch)
        z_mu = jnp.mean(z_obs, axis=0)
        z_diff = self.conv_flag+1.

        if step == "while":
            z_mu, z_diff, Wk, Vk, idx = lax.while_loop(self.cond_fun, 
                                                       self.while_step, 
                                                       init_val=(z_mu, z_diff, Wk, Vk, 0),
                                                       )
        elif step == "for":
            _, (z_mu, z_diff, Wk, Vk, conv_idx) = lax.scan(self.for_step, 
                                                           init=(z_mu, z_diff, Wk, Vk, 0),
                                                           xs=self.subkeys,
                                                           )
            
            idx = self.max_iter
            
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return z_mu, None, z_diff, idx


#%% GEORCE_FM Step

class GEORCE_FM_Step(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 iters:int=1000,
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.iters = jnp.ones(iters)
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
               z_mu:Array,
               *args,
               )->Array:

        zt = zt.reshape(self.N, -1, self.dim)
        
        path_energy = vmap(self.path_energy, in_axes=(0,0,0,None))(self.z_obs, zt, self.G0, z_mu)
        
        return jnp.sum(self.wi*path_energy)
    
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
        dmu = 2.*jnp.mean(jnp.einsum('...ij,i->...j', Gt[:,-1], z_mu) - \
                          jnp.einsum('...ij,...j->...i', Gt[:,-1], zt[:,-1]), axis=0)

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
    
    def Wk(self,
           ginv_sum_inv:Array,
           )->Array:

        lhs = jnp.einsum('t,tij->tij', self.wi, ginv_sum_inv)
        
        return jnp.sum(lhs, axis=0)
    
    def Vk(self,
           gt_inv:Array,
           ginv_sum_inv:Array,
           g_cumsum:Array,
           ):
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', gt_inv[:,:-1], g_cumsum), axis=1),
                            )
        
        return jnp.sum(rhs, axis=0)
    
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
                       Wk:Array,
                       Vk:Array,
                       )->Array:

        return jnp.linalg.solve(Wk,Vk)
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  z_mu:Array,
                  z_mu_hat:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        zt_new = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(alpha*ut_hat+(1-alpha)*ut, axis=1)
        z_mu_new = alpha*z_mu_hat+(1.-alpha)*z_mu

        return (zt_new, z_mu_new)

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, z_mu, gt, gt_inv, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple[Array,Array],
                    idx:int,
                    )->Array:
        
        zt, ut, z_mu, gt, gt_inv, Wk, Vk, grad_norm = carry
        
        g_cumsum = jnp.cumsum(gt[:,::-1], axis=1)[:,::-1]
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        Wk = self.Wk(ginv_sum_inv)
        Vk = self.Vk(gt_inv, ginv_sum_inv, g_cumsum)
        
        z_mu_hat = self.frechet_update(Wk, Vk)
        mut = self.curve_update(z_mu_hat, g_cumsum, gt_inv, ginv_sum_inv)

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ut[:,:-1], axis=1)
        
        gt, Gt = self.gt(zt, ut[:,1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim),  jnp.linalg.inv(Gt)),
                                 axis=1)
        
        grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu, Gt, gt))
        
        return ((zt, ut, z_mu, gt, gt_inv, Wk, Vk, grad_norm),)*2

    def __call__(self, 
                 z_obs:Array,
                 wi:Array,
                 G0:Array,
                 Ginv0:Array,
                 )->Array:

        self.z_obs = z_obs
        self.wi = wi
        z_mu_init = jnp.mean(z_obs, axis=0)
        self.N, self.dim = self.z_obs.shape
        
        self.G0 = G0
        self.Ginv0 = Ginv0

        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Denergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        zt = self.init_curve(self.z_obs, z_mu_init)
        ut = jnp.ones((self.N, self.T, self.dim), dtype=z_obs.dtype)*(z_mu_init-self.z_obs.reshape(-1,1,self.dim))/self.T
        
        gt, Gt = self.gt(zt, ut[:,1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim),  jnp.linalg.inv(Gt)),
                                 axis=1)
        grad_norm = jnp.linalg.norm(self.Denergy_frechet(zt, ut, z_mu_init, Gt, gt))
        
        Wk = jnp.zeros((self.dim, self.dim), dtype=z_obs.dtype)
        Vk = jnp.zeros(self.dim, dtype=z_obs.dtype)

        val, _ = lax.scan(self.georce_step,
                          init=(zt, ut, z_mu_init, gt, gt_inv, Wk, Vk, grad_norm),
                          xs=self.iters,
                          )

        Wk = val[-3]
        Vk = val[-2]
        grad_norm = val[-1]
            
        return Wk, Vk, grad_norm