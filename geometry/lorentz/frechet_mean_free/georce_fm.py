#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.lorentz.manifolds import LorentzFinslerManifold
from geometry.line_search import Backtracking

#%% Frechet Mean using GEORCE_FM

class GEORCE_FM(ABC):
    def __init__(self,
                 M:LorentzFinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {},
                 parallel:bool=True,
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if parallel:
            self.energy = self.vmap_energy
            self.gs = self.vmap_gs
            self.hs = self.vmap_hs
            self.rs = self.vmap_rs
        else:
            self.energy = self.loop_energy
            self.gs = self.loop_gs
            self.hs = self.loop_hs
            self.rs = self.loop_rs
        
        self.Lt = grad(M.F, argnums=0)
        self.Lz = jacfwd(M.F, argnums=1)
        self.Lu = jacfwd(M.F, argnums=2)
        
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
        
        zs = self.init_fun(z0, zT, self.T)
        total = jnp.vstack((z0, zs, zT))
        us = total[1:]-total[:-1]
        
        return zs, us
    
    def vmap_energy(self, 
                    zs:Array,
                    z_mu:Array,
                    *args,
                    )->Array:
        
        zs = zs.reshape(self.N, -1, self.dim)
        
        energy = vmap(self.path_energy, in_axes=(0,0,None))(self.z_obs, zs, z_mu)

        return jnp.sum(self.wi*energy)
    
    def loop_energy(self, 
                    zs:Array,
                    z_mu:Array,
                    *args,
                    )->Array:
            
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w = y
            
            energy += w*self.path_energy(z_obs, z, z_mu)

            return (energy,)*2
        
        zs = zs.reshape(self.N, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zs, self.z_obs, self.wi),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zs:Array,
                    z_mu:Array,
                    )->Array:
        
        us = jnp.vstack((zs[0]-z0,
                         zs[1:]-zs[:-1],
                         z_mu-zs[-1],
                         ))

        ts = self.update_ts(z0, zs, us)
        
        val1 = self.M.F(self.t0, z0, -us[0])**2
        val2 = vmap(lambda t,x,v: self.M.F(t,x,v)**2)(ts[:-1], zs, -us[1:])

        return val1+jnp.sum(val2)
    
    def Denergy(self,
                zs:Array,
                *args,
                )->Array:

        return grad(self.energy, argnums=0)(zs, *args)/self.N
    
    def Denergy_frechet(self,
                        zs:Array,
                        z_mu,
                        )->Array:

        dcurve, dmu = grad(self.energy, argnums=(0,1))(zs, z_mu)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))/self.N
    
    def vmap_inner_product(self,
                      ts:Array,
                      zs:Array,
                      us:Array,
                      )->Array:
        
        Gs = vmap(vmap(self.M.G))(ts,zs,-us)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', us, Gs, us))
    
    def vmap_inner_product_h(self,
                        ts:Array,
                        zs:Array,
                        u0:Array,
                        us:Array,
                        )->Array:
        
        Gs = vmap(vmap(self.M.G))(ts, zs,-us)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', u0, Gs, u0)), Gs
    
    def loop_inner_product(self,
                      ts:Array,
                      zs:Array,
                      us:Array,
                      )->Array:
        
        Gs = vmap(self.M.G)(ts,zs,-us)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', us, Gs, us))
    
    def loop_inner_product_h(self,
                        ts:Array,
                        zs:Array,
                        u0:Array,
                        us:Array,
                        )->Array:
        
        Gs = vmap(self.M.G)(ts, zs,-us)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', u0, Gs, u0)), Gs
    
    def vmap_gs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        gs = lax.stop_gradient(grad(self.vmap_inner_product, argnums=1)(ts, zs, us))

        return gs
    
    def vmap_hs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        hs, Gs = lax.stop_gradient(grad(self.vmap_inner_product_h, argnums=3, has_aux=True)(ts, zs, us, us))
        
        return hs, Gs
    
    def vmap_rs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        rs = lax.stop_gradient(grad(self.vmap_inner_product, argnums=0)(ts, zs, us))

        return rs
    
    def loop_gs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        def step_gs(g:Array,
                    y:Tuple,
                    )->Tuple:
            
            t,z,u = y
            
            g = lax.stop_gradient(grad(self.loop_inner_product, argnums=1)(t,z,u))
            
            return (g,)*2
        
        _, gs = lax.scan(step_gs,
                         init=jnp.zeros((self.T-1, self.dim), dtype=zs.dtype),
                         xs=(ts,zs,us),
                         )

        return gs
    
    def loop_hs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        def step_hs(c:Tuple,
                    y:Tuple,
                    )->Tuple:
            
            t,z,u = y
            
            h, G = lax.stop_gradient(grad(self.loop_inner_product_h, argnums=3, has_aux=True)(t, z, u, u))
            
            return ((h,G),)*2
        
        _, (hs, Gs) = lax.scan(step_hs,
                               init=(jnp.zeros((self.T, self.dim), dtype=zs.dtype),
                                     jnp.zeros((self.T, self.dim,self.dim), dtype=zs.dtype)),
                               xs=(ts,zs,us),
                               )
        
        return hs, Gs
    
    def loop_rs(self,
                ts:Array,
                zs:Array,
                us:Array,
                )->Array:
        
        def step_rs(g:Array,
                    y:Tuple,
                    )->Tuple:
            
            t,z,u = y
            
            r = lax.stop_gradient(grad(self.loop_inner_product, argnums=0)(t,z,u))
            
            return (r,)*2
        
        _, rs = lax.scan(step_rs,
                         init=jnp.zeros(self.T-1, dtype=zs.dtype),
                         xs=(ts,zs,us),
                         )

        return rs
    
    def pi(self,
           rs:Array,
           Lts:Array,
           )->Array:
        
        def step(pis:Array,
                 step:Tuple[Array,Array],
                 )->Tuple[Array, Array]:
            
            rs, Ls = step
            
            return ((rs+pis*Ls+pis),)*2
        
        _, pi = lax.scan(step,
                         xs=(rs[::-1], Lts[::-1]),
                         init=0.0,
                         )
        
        return jnp.hstack((pi[::-1], 0.0)).reshape(-1,1)
    
    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     gs_inv:Array,
                     ginv_sum_inv:Array,
                     hs:Array,
                     pis:Array,
                     Lus:Array,
                     )->Array:
        
        diff = jnp.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = jnp.sum(jnp.einsum('...ij,...j->...i', gs_inv, g_cumsum+hs+pis*Lus), axis=1)+2.0*diff

        muT = -jnp.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mus = muT+g_cumsum+hs+pis*Lus
        
        return mus
    
    def frechet_update(self,
                       g_cumsum:Array,
                       gs_inv:Array,
                       ginv_sum_inv:Array,
                       hs:Array,
                       pis:Array,
                       Lus:Array,
                       )->Array:
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', gs_inv, g_cumsum+hs+pis*Lus), axis=1),
                            )
            
        lhs = jnp.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = jnp.linalg.solve(jnp.sum(lhs, axis=0), 
                              jnp.sum(rhs, axis=0),
                              )
        
        return mu
    
    def update_ts(self,
                  z0:Array,
                  zs:Array,
                  us:Array,
                  )->Array:
        
        def step(t:Array,
                 step:Tuple[Array,Array],
                 )->Array:
            
            z, dz = step
            
            t += self.M.F(t, z, -dz)
            
            return (t,)*2
        
        zs = jnp.vstack((z0, zs))
        
        _, ts = lax.scan(step,
                         init=self.t0,
                         xs = (zs, us),
                         )
        
        return ts
    
    def update_xs(self,
                  zs:Array,
                  alpha:Array,
                  z_mu:Array,
                  z_mu_hat:Array,
                  us_hat:Array,
                  us:Array,
                  )->Array:
        
        zs_new = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(alpha*us_hat+(1-alpha)*us, axis=1)
        z_mu_new = alpha*z_mu_hat+(1.-alpha)*z_mu

        return zs_new, z_mu_new

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        ts, zs, us, z_mu, rs, hs, gs, gs_inv, grad_norm, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        ts, zs, us, z_mu, rs, hs, gs, gs_inv, grad_norm, idx = carry
        
        Lts = vmap(vmap(self.Lt))(ts[:,:-1],zs,us[:,1:])
        Lzs = vmap(vmap(self.Lz))(ts[:,:-1],zs,us[:,1:])
        Lus = jnp.concatenate((vmap(self.Lu, in_axes=(None,0,0))(self.t0, self.z_obs, us[:,0]).reshape(self.N,1,-1),
                               vmap(vmap(self.Lu))(ts[:,:-1], zs, us[:,1:]),
                               ), 
                              axis=1,
                              )
        pis = vmap(self.pi)(rs, Lts).reshape(self.N,-1,1)
        g_cumsum = jnp.concatenate((jnp.cumsum((gs+pis[:,1:]*Lzs)[:,::-1], axis=1)[:,::-1], 
                                    jnp.zeros((self.N, 1, self.dim))), 
                                   axis=1)
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gs_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gs_inv, ginv_sum_inv, hs, pis, Lus)
        mus = self.curve_update(z_mu_hat, g_cumsum, gs_inv, ginv_sum_inv, hs, pis, Lus)

        us_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gs_inv, mus)
        tau = self.line_search(zs, z_mu, z_mu_hat, us_hat, us)

        us = tau*us_hat+(1.-tau)*us
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zs = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(us[:,:-1], axis=1)
        ts = vmap(self.update_ts)(self.z_obs, zs, us)
        
        gs = self.gs(ts[:,:-1], zs, us[:,1:])
        rs = self.rs(ts[:,:-1], zs, us[:,1:])
        hs, Gs = self.hs(jnp.concatenate((jnp.ones((self.N,1))*self.t0, ts[:,:-1]), axis=1),
                         jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zs), axis=1), 
                         us)
        gs_inv = jnp.linalg.inv(Gs)
        
        grad_norm = jnp.linalg.norm(self.Denergy_frechet(zs, z_mu))
        
        return (ts, zs, us, z_mu, rs, hs, gs, gs_inv, grad_norm, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        ts, zs, us, z_mu = carry
        
        Lts = vmap(vmap(self.Lt))(ts[:,:-1],zs,us[:,1:])
        Lzs = vmap(vmap(self.Lz))(ts[:,:-1],zs,us[:,1:])
        Lus = jnp.concatenate((vmap(self.Lu, in_axes=(None,0,0))(self.t0, self.z_obs, us[:,0]).reshape(self.N,1,-1),
                               vmap(vmap(self.Lu))(ts[:,:-1], zs, us[:,1:]),
                               ), 
                              axis=1,
                              )
        
        gs = self.gs(ts[:,:-1], zs, us[:,1:])
        rs = self.rs(ts[:,:-1], zs, us[:,1:])
        hs, Gs = self.hs(jnp.concatenate((jnp.ones((self.N,1))*self.t0, ts[:,:-1]), axis=1),
                         jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zs), axis=1), 
                         us)
        gs_inv = jnp.linalg.inv(Gs)
        
        pis = vmap(self.pi)(rs, Lts).reshape(self.N, -1, 1)
        
        g_cumsum = jnp.concatenate((jnp.cumsum((gs+pis[:,1:]*Lzs)[:,::-1], axis=1)[:,::-1], 
                                    jnp.zeros((self.N, 1, self.dim))), 
                                   axis=1)
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(gs_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, gs_inv, ginv_sum_inv, hs, pis, Lus)
        mus = self.curve_update(z_mu_hat, g_cumsum, gs_inv, ginv_sum_inv, hs, pis, Lus)

        us_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gs_inv, mus)
        tau = self.line_search(zs, z_mu, z_mu_hat, us_hat, us)

        us = tau*us_hat+(1.-tau)*us
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zs = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(us[:,:-1], axis=1)
        ts = vmap(self.update_ts)(self.z_obs, zs, us)
        
        return ((ts, zs, us, z_mu),)*2

    def __call__(self, 
                 t0:Array,
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:
        
        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_xs,
                                        grad_fun = lambda z,*args: self.Denergy(z,*args).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.t0 = t0
        self.z_obs = z_obs
        self.N, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zs, us = vmap(self.init_curve, in_axes=(0,None))(self.z_obs, z_mu_init)
        ts = vmap(self.update_ts)(self.z_obs, zs, us)
        
        if step == "while":
            gs = self.gs(ts[:,:-1], zs, us[:,1:])
            rs = self.rs(ts[:,:-1], zs, us[:,1:])
            hs, Gs = self.hs(jnp.concatenate((jnp.ones((self.N,1))*self.t0, ts[:,:-1]), axis=1),
                             jnp.concatenate((self.z_obs.reshape(-1,1,self.dim), zs), axis=1), 
                             us)
            gs_inv = jnp.linalg.inv(Gs)
            grad_norm = jnp.linalg.norm(self.Denergy_frechet(zs, z_mu_init))
            
            s, zs, us, z_mu, rs, hs, gs, gs_inv, grad_norm, idx = lax.while_loop(self.cond_fun, 
                                                                                 self.while_step, 
                                                                                 init_val=(ts, zs, us, z_mu_init, 
                                                                                           rs, hs, gs, gs_inv, grad_norm, 0),
                                                                                 )
            
            ts = ts[:,::-1]
            zs = zs[:,::-1]
            
        elif step == "for":
                
            _, (ts, zs, us, z_mu) = lax.scan(self.for_step,
                                             init=(ts, zs, us, z_mu_init),
                                             xs=jnp.ones(self.max_iter),
                                             )

            grad_norm = None
            idx = self.max_iter
            
            ts = ts[:,:,::-1]
            zs = zs[:,:,::-1]
            
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return z_mu, ts, zs, grad_norm, idx