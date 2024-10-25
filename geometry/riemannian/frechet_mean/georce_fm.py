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

class GEORCE_FM(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_method:str="soft",
                 line_search_params:Dict = {},
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
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
        
        return jnp.sum(path_energy)
    
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
    
    def path_energy(self, 
                    z0:Array,
                    zt:Array,
                    G0:Array,
                    )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        return val1+jnp.sum(val2)
    
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
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                *args,
                )->Array:

        return grad(self.energy, argnums=0)(zt)
    
    def Denergy_frechet(self,
                        zt:Array,
                        z_mu:Array,
                        )->Array:
        
        #grad_zt, grad_z_mu = grad(self.energy_frechet, argnums=(0,1))(zt, z_mu)
        
        #return jnp.hstack((grad_zt.reshape(-1), grad_z_mu))

        return grad(self.energy_frechet, argnums=1)(zt, z_mu)
    
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
                  z_mu:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        add_term = jnp.cumsum(alpha*ut_hat+(1-alpha)*ut, axis=1)
        
        return vmap(lambda z, add: z+add)(self.z_obs, add_term)

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, z_mu, gt, gt_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, z_mu, gt, gt_inv, grad, idx = carry
        
        z_mu_hat = self.frechet_update(gt, gt_inv)
        
        mut = vmap(self.unconstrained_opt, in_axes=(0,None,0,0,0))(self.z_obs, 
                                                                   z_mu_hat, 
                                                                   gt, 
                                                                   gt_inv,
                                                                   self.wi,
                                                                   )

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = vmap(lambda z, add: z+add)(self.z_obs,
                                        jnp.cumsum(ut[:,:-1], axis=1),
                                        )

        gt = vmap(lambda z,u: self.gt(z,u[1:]))(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim), 
                                  (vmap(lambda z: self.M.Ginv(z))(zt.reshape(-1,self.dim))).reshape(self.N, 
                                                                                                    -1, 
                                                                                                    self.dim,
                                                                                                    self.dim)),
                                 axis=1)
        grad = self.Denergy_frechet(zt,z_mu)
        
        return (zt, ut, z_mu, gt, gt_inv, grad, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut, z_mu = carry
        
        gt = vmap(lambda z,u: self.gt(z,u[1:]))(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim), 
                                  (vmap(lambda z: self.M.Ginv(z))(zt.reshape(-1,self.dim))).reshape(self.N, 
                                                                                                    -1, 
                                                                                                    self.dim,
                                                                                                    self.dim)),
                                 axis=1)
        
        z_mu_hat = self.frechet_update(gt, gt_inv)
        
        mut = vmap(self.unconstrained_opt, in_axes=(0,None,0,0,0))(self.z_obs, 
                                                                   z_mu_hat, 
                                                                   gt, 
                                                                   gt_inv,
                                                                   self.wi,
                                                                   )

        ut_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, gt_inv, mut)
        tau = self.line_search(zt, z_mu_hat, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        z_mu = tau*z_mu_hat+(1.-tau)*z_mu
        zt = vmap(lambda z, add: z+add)(self.z_obs,
                                        jnp.cumsum(ut[:,:-1], axis=1),
                                        )
        
        return ((zt, ut, z_mu),)*2
    
    def unconstrained_opt(self, 
                          z:Array,
                          mu:Array,
                          gt:Array, 
                          gt_inv:Array,
                          wi:Array,
                          )->Array:
        
        diff = wi*(mu-z)
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def frechet_update(self,
                       gt:Array,
                       gt_inv:Array,
                       )->Array:
        
        gt = gt.reshape(self.N, -1, self.dim)
        gt_inv = gt_inv.reshape(self.N, -1, self.dim, self.dim)
        
        g_cumsum = jnp.cumsum(gt[:,::-1], axis=1)[:,::-1]
        ginv_sum = jnp.linalg.inv(jnp.sum(gt_inv, axis=1))
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', gt_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        mu = jnp.linalg.solve(jnp.sum(ginv_sum, axis=0), 
                              jnp.sum(rhs, axis=0),
                              )
        
        return mu

    def __call__(self, 
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 step:str="while",
                 )->Array:

        self.z_obs = z_obs.astype('float64')
        
        self.dtype = self.z_obs.dtype
        self.N, self.dim = self.z_obs.shape
        self.G0 = vmap(self.M.G)(self.z_obs)
        self.Ginv0 = jnp.linalg.inv(self.G0)
        
        if wi is None:
            self.wi = jnp.ones(self.N)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = self.z_obs[0]#jnp.mean(self.z_obs, axis=0)

        zt = self.init_curve(self.z_obs, z_mu_init)
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_xt,
                                            grad_fun = lambda z,*args: self.Denergy(z,*args).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_xt,
                                         **self.line_search_params,
                                         )
        
        ut = jnp.ones((self.T, self.N, self.dim), dtype=self.dtype)*(z_mu_init-self.z_obs)/self.T
        ut = ut.reshape(self.N, self.T, self.dim)
        
        if step == "while":
            gt = vmap(lambda z,u: self.gt(z,u[1:]))(zt,ut)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gt_inv = jnp.concatenate((self.Ginv0.reshape(self.N, -1, self.dim, self.dim), 
                                      (vmap(lambda z: self.M.Ginv(z))(zt.reshape(-1,self.dim))).reshape(self.N, 
                                                                                                        -1, 
                                                                                                        self.dim,
                                                                                                        self.dim)),
                                     axis=1)
            grad = self.Denergy_frechet(zt, z_mu_init)
            
            zt, _, z_mu, _, _, grad, idx = lax.while_loop(self.cond_fun, 
                                                          self.while_step, 
                                                          init_val=(zt, ut, z_mu_init, gt, gt_inv, grad, 0))
            
            dist = self.length_frechet(zt, z_mu)
            
            #zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut, z_mu_init),
                              xs=jnp.ones(self.max_iter),
                              )
            
            z_mu = val[2]
            zt = val[0]
            grad = vmap(self.Denergy_frechet)(zt, z_mu)
            dist = vmap(self.length_frechet)(zt, z_mu)
            #zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return z_mu, zt, dist, grad, idx

        