#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:52:51 2025

@author: fmry
"""

#%% Modules

from geometry.setup import *

from typing import Literal, List

from geometry.riemannian.manifolds import RiemannianManifold
from geometry.line_search import NaiveBacktracking
from geometry.riemannian.frechet_mean import GEORCE_FM
from geometry.riemannian.geodesics import GEORCE

#%% Class Principal Geodesic Analysis

class PGA(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 index_step:int=1,
                 index_tol:float=1e-4,
                 rho:float=0.5,
                 epsilon:float=1e-6,
                 parallel:bool=True,
                 pga_method:Literal["approximation", "exact"] = "exact",
                 orthgornal_method:Literal["tangent_space", "projection"] = "tangent_space",
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        self.index_step = index_step
        self.index_tol = index_tol
        
        self.rho = rho
        self.epsilon = epsilon
        self.pga_method = pga_method
        self.orthgornal_method = orthgornal_method
        
        self.seed = seed
        self.key = jrandom.key(seed)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.geodesic = GEORCE(M = M,
                               init_fun=init_fun,
                               T=T,
                               max_iter=max_iter,
                               line_search_method="soft",
                               line_search_params = {'rho': rho},
                               )
            
        self.frechet_mean = GEORCE_FM(M=M,
                                      init_fun=init_fun,
                                      T=T,
                                      tol=tol,
                                      max_iter=max_iter,
                                      line_search_params = {'rho': rho},
                                      parallel=parallel,
                                      )
        
        self.single_exact_pga = SinglePrincipalGeodesic(M=M,
                                                        init_fun=init_fun,
                                                        T=T,
                                                        tol=tol,
                                                        max_iter=max_iter,
                                                        index_step=index_step,
                                                        index_tol=index_tol,
                                                        rho=rho,
                                                        epsilon=epsilon,
                                                        parallel=parallel,
                                                        )
        
        
        self.z_obs = None
        self.N, self.dim = None, None
        self.n_components = None
        
        self.z_mu = None
        self.G_mu = None
        self.log_obs = None
        
        self.var_explained = None 
        self.principal_zgeo_backward = None 
        self.principal_zgeo_forward = None 
        self.V = None
        
        return
    
    def __str__(self,
                )->str:
        
        return "Principal Geodesic Analysis object"
    
    def _transform_tangent_space_orthorgonal(self,
                                             V:Array,
                                             )->Array:
        
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        else:
            V = V.T
            
        norms = jnp.sqrt(jnp.einsum('ij,jk,ik->i', V.T, self.G_mu, V.T))  # shape (d,)
        V = V / norms
            
        G_w = jnp.einsum('ij,bj->bi', self.G_mu, self.log_obs)      # G w_i for each batch
        Vt_G_w = jnp.einsum('ij,bj->bi', V.T, G_w)                  # g(w_i, v_j) for each i,j
        proj = jnp.einsum('ij,bi->bj', V, Vt_G_w)                   # V × coefficients
        u0_complement = self.log_obs - proj
        
        z_obs_complement = jnp.stack([self.M.Exp_ode(self.z_mu, u)[-1] for u in u0_complement])
        
        return z_obs_complement
    
    def _transform_projection_orthorgonal(self,
                                          zt_pg_curve:Array,
                                          u0_proj:Array,
                                          )->Array:

        u0_proj *= self.T

        u0_complement = [self.M.parallel_transport_ode(curve, v)[1][-1] for curve,v in zip(zt_pg_curve, u0_proj)]
        z_obs_complement = jnp.stack([self.M.Exp_ode(self.z_mu, u0)[-1] for u0 in u0_complement])
        
        return z_obs_complement
    
    def _approximate_pga(self,
                         )->Tuple[Array, Array, Array, Array]:
        
        pga_S = jnp.mean(jnp.einsum('...i,...j->...ij', self.log_obs, self.log_obs), axis=0)
        U, S, V = jnp.linalg.svd(pga_S)
        V = V[:self.n_components]
        
        var_explained = (S*S/(S*S).sum())[:self.n_components]
        principal_zgeo_backward = vmap(self.M.Exp_ode, in_axes=(None, 0, None))(self.z_mu, -V, self.T)
        principal_zgeo_forward = vmap(self.M.Exp_ode, in_axes=(None, 0, None))(self.z_mu, V, self.T)
        
        return var_explained, principal_zgeo_backward, principal_zgeo_forward, V
    
    def _update_single_pga(self,
                           idx:int,
                           z_obs:Array,
                           V:Array,
                           var_explained:Array,
                           )->Array:
        
        si, pi, zit_proj, zt_pg_backward, zt_pg_forward, uit_proj, ut_pg_backward, ut_pg_forward, *_ \
            = self.single_exact_pga(self.z_mu, z_obs)
            
        zt_pg_curve = [jnp.vstack((self.z_mu, zt_pg_backward[:(p+1)])) if s == 0 \
                       else jnp.vstack((self.z_mu, zt_pg_forward[:(p+1)])) for s,p in zip(si,pi)]
            
        var_explained = var_explained.at[idx].set(
            jnp.sum(jnp.stack([self.M.length(curve) for curve in zt_pg_curve]))
            )
            
        v = ut_pg_forward[0]*self.T
        v /= jnp.sqrt(jnp.einsum('i,ij,j->', v, self.G_mu, v))
        V = V.at[idx].set(v)

        if self.orthgornal_method == "projection":
            u0_proj  = -uit_proj[:,-1]
            z_obs_complement = self._transform_projection_orthorgonal(zt_pg_curve, u0_proj)
        elif self.orthgornal_method == "tangent_space":
            z_obs_complement = self._transform_tangent_space_orthorgonal(V[:(idx+1)])
            
        return z_obs_complement, V, var_explained, zt_pg_backward, zt_pg_forward
    
    def _exact_pga(self,
                   )->Tuple[Array, Array, Array, Array]:
        
        z_obs = self.z_obs
        V = jnp.zeros((self.n_components, self.dim))
        var_explained = jnp.zeros(self.n_components)
        principal_zgeo_backward, principal_zgeo_forward = [], []
        for idx in range(self.n_components):
            z_obs, V, var_explained, zt1, zt2 = self._update_single_pga(idx, z_obs, V, var_explained)
            principal_zgeo_backward.append(zt1)
            principal_zgeo_forward.append(zt2)
        principal_zgeo_backward, principal_zgeo_forward = jnp.stack(principal_zgeo_backward), jnp.stack(principal_zgeo_forward)

        var_explained /= jnp.sum(var_explained)

        return var_explained, principal_zgeo_backward, principal_zgeo_forward, V
    
    def get_principal_geodesics(self,
                                )->Tuple[Array, Array]:
        
        if (self.principal_zgeo_backward is not None) and (self.principal_zgeo_forward is not None):        
            return self.principal_zgeo_backward, self.principal_zgeo_forward
        else:
            raise ValueError("The principal geodesics have not been computed")
            
    def get_variance_explained(self,
                               )->Array:
        
        if self.var_explained is not None:
            return self.var_explained
        else:
            raise ValueError("The principal geodesics have not been computed")
    
    def fit(self,
            z_obs:Array,
            n_components:int=1,
            )->None:
        
        self.z_obs = z_obs
        self.N, self.dim = z_obs.shape
        self.n_components = n_components if n_components > self.dim else self.dim
        
        self.z_mu, zt, *_ = self.frechet_mean(self.z_obs)
        self.G_mu = self.M.G(self.z_mu)
        self.log_obs = (zt[:,1] - zt[:,0])*self.T
        
        if self.pga_method == "approximation":
            self.var_explained, self.principal_zgeo_backward, self.principal_zgeo_forward, self.V \
                = self._approximate_pga()
        elif self.pga_method == "exact":
            self.var_explained, self.principal_zgeo_backward, self.principal_zgeo_forward, self.V \
                = self._exact_pga()
            
        return
    
    def sample(self,
               n_samples:int=100,
               n_components:int=1,
               T:int=100
               )->Array:
        
        if self.n_components > n_components:
            raise ValueError("The number of principal geodesics is less than the computed")
        if self.z_mu is None:
            raise ValueError("The Frechet mean has not been computed")
        if self.V is None:
            raise ValueError("The principal geodesics have not been computed")
            
        self.key, subkey = jrandom.split(self.key)
        
        coef = jrandom.normal(subkey, shape(n_samples, n_components))
        samples = jnp.stack([self.M.Exp_ode(self.z_mu, jnp.einsum('i,ij->j', c, self.V[:n_components]), T) for c in coef])
        
        return samples

#%% Class Principal Geodesic Analysis

class SinglePrincipalGeodesic(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 index_step:int=1,
                 index_tol:float=1e-4,
                 rho:float=0.5,
                 epsilon:float=1e-6,
                 parallel:bool=True,
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        self.index_step = index_step
        self.index_tol = index_tol
        
        self.rho = rho
        self.epsilon = epsilon
        
        if parallel:
            self.target_energy = self.vmap_target_funciton
            self.gt = self.vmap_gt
        else:
            self.target_energy = self.loop_energy
            self.gt = self.loop_target_funciton
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.pca = PCA()
        
        return
    
    def __str__(self,
                )->str:
        
        return "Principal Geodesic Analysis object"
    
    def init_pg(self, 
                z0:Array, 
                v:Array,
                )->Array:
        
        ut = v*jnp.ones((self.T, self.dim))
        zt = z0 + jnp.cumsum(ut, axis=0)
        
        return zt, ut
    
    def init_proj(self, 
                  z0:Array, 
                  zT:Array,
                  )->Array:
        
        zt = self.init_fun(z0, zT, self.T)
        total = jnp.vstack((z0, zt, zT))
        ut = total[1:]-total[:-1]
        
        return zt, ut
    
    def init_projection_points(self,
                               zt_pg_backward:Array,
                               zt_pg_forward:Array,
                               )->Tuple[Array,Array]:
        
        # Expand z0 to (n, 1, d) to broadcast over T
        z0_expanded = jnp.expand_dims(self.z_obs, axis=1)  # (n, 1, d)
    
        # Compute squared Euclidean distances
        dist_backward = jnp.sum((zt_pg_backward - z0_expanded) ** 2, axis=-1)  # (n, T)
        dist_forward  = jnp.sum((zt_pg_forward  - z0_expanded) ** 2, axis=-1)  # (n, T)
    
        # Concatenate distances along time axis
        dists = jnp.concatenate([dist_backward, dist_forward], axis=1)  # (n, 2T)
    
        # Find index of minimum distance
        P_total = jnp.argmin(dists, axis=1)  # (n,)
    
        # Determine S: -1 if from backward (P < T), +1 if from forward (P >= T)
        si = jnp.where(P_total < self.T, 0, 1)

        # Compute time index within respective array
        pi = jnp.where(P_total < self.T, P_total, P_total - self.T)
    
        return si, pi
        
    def update_projection_points(self,
                                 si:Array,
                                 pi:Array,
                                 zt_pg_backward:Array,
                                 zt_pg_forward:Array,
                                 uit_proj:Array,
                                 Git_proj:Array,
                                 git_proj:Array,
                                 Git_proj_inv:Array,
                                 Gi_proj_inv_sum:Array,
                                 )->Tuple[Array,Array]:
        
        zT_proj = self.get_projection(si, pi, zt_pg_backward, zt_pg_forward)
        
        pi_new = jnp.where(pi < self.T, pi + self.index_step, pi - self.index_step)
        si_new = jnp.where(pi_new < 0, -si, si)
        pi_new = jnp.where(pi_new < 0, -pi_new, pi_new)
        
        zT_proj_new = self.get_projection(si_new, pi_new, zt_pg_backward, zt_pg_forward)
        
        delta_xT = zT_proj_new - zT_proj
        delta_muT = -vmap(jnp.linalg.solve)(Gi_proj_inv_sum, 2*delta_xT)
        delta_ut = -0.5*jnp.einsum('...tij,...j->...ti', Git_proj_inv, delta_muT)
        delta_xt = jnp.cumsum(delta_ut, axis=1)[:,:-1]
        
        term1 = 2.*jnp.einsum('...tij,...ti,...tj->...t', Git_proj, uit_proj, delta_ut)
        term2 = jnp.einsum('...ti,...ti->...t', git_proj, delta_xt)
        
        delta_energy = jnp.sum(term1, axis=1) + jnp.sum(term2, axis=1)
        
        pi_new = jnp.where((jnp.abs(delta_energy) > self.index_tol) & (delta_energy > 0) & (pi < self.T),
                           pi - self.index_step,#delta_energy,
                           pi,
                           )
        pi_new = jnp.where((jnp.abs(delta_energy) > self.index_tol) & (delta_energy < 0) & (pi < self.T),
                           pi_new + self.index_step,#self.index_scale,
                           pi_new,
                           )
        
        pi_new = jnp.where((jnp.abs(delta_energy) > self.index_tol) & (delta_energy > 0) & (pi > self.T),
                           pi_new + self.index_step,#delta_energy,
                           pi_new,
                           )
        pi_new = jnp.where((jnp.abs(delta_energy) > self.index_tol) & (delta_energy < 0) & (pi > self.T),
                           pi_new - self.index_step,#delta_energy,
                           pi_new,
                           )

        si_new = jnp.where(pi_new < 0, -si, si)
        pi_new = jnp.where(pi_new < 0, -pi_new, pi_new)
        
        return si_new, pi_new
    
    def get_projection(self,
                       si:Array, 
                       pi:Array, 
                       zt_pg_backward:Array, 
                       zt_pg_forward:Array,
                       )->Array:
        
        zt_pg_backward = jnp.vstack((self.z_mu, zt_pg_backward))
        zt_pg_forward = jnp.vstack((self.z_mu, zt_pg_forward))
    
        # Gather from backward and forward
        z_back = zt_pg_backward[pi]  # shape (n, d)
        z_fwd  = zt_pg_forward[pi]   # shape (n, d)
    
        # Choose using S: if S == -1 => z_back, if S == 1 => z_fwd
        zT_proj = jnp.where(si[:, None] == 0, z_back, z_fwd)  # shape (n, d)
    
        return zT_proj
    
    def vmap_target_funciton(self, 
                             zit_proj:Array,
                             zt_pg_backward:Array,
                             zt_pg_forward:Array,
                             si:Array,
                             pi:Array,
                             *args,
                             )->Array:

        zT_proj = self.get_projection(si, pi, zt_pg_backward, zt_pg_forward)

        energy1 = vmap(self.path_target_funciton)(self.G0_proj.squeeze(), 
                                                  self.z_obs, 
                                                  zit_proj, 
                                                  zT_proj,
                                                  )

        return jnp.sum(energy1)
    
    def loop_target_funciton(self, 
                             zit_proj:Array,
                             zt_pg_backward:Array,
                             zt_pg_forward:Array,
                             si:Array,
                             pi:Array,
                             *args,
                             )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple[Array, Array, Array, Array],
                        )->Tuple:
            
            G0i, z0i, zit, ziT = y
            
            energy += jnp.sum(vmap(self.path_target_funciton)(G0i, z0i, zit, ziT))

            return (energy,)*2
        
        zT_proj = self.get_projection(si, pi, zt_pg_backward, zt_pg_forward)
        
        energy1, _ = lax.scan(step_energy,
                              init=0.0,
                              xs=(self.G0_proj.squeeze(), self.z_obs, zit_proj, zT_proj),
                              )

        return energy1
    
    def path_target_funciton(self, 
                             G0:Array,
                             z0:Array,
                             zt:Array,
                             zT:Array,
                             )->Array:
        
        term1 = zt[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(self.M.G)(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1 + jnp.sum(val2) + val3
    
    def pg_inner_product(self,
                         zt:Array,
                         ut:Array,
                         )->Array:
            
        Gt = vmap(self.M.G)(zt)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut)), Gt
    
    def vmap_inner_product(self,
                           zt:Array,
                           ut:Array,
                           )->Array:
            
        Gt = vmap(vmap(self.M.G))(zt)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut)), Gt
    
    def loop_inner_product(self,
                           zt:Array,
                           ut:Array,
                           )->Array:
            
        Gt = vmap(self.M.G)(zt)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ut, Gt, ut)), Gt
    
    def vmap_gt(self,
                zt:Array,
                ut:Array,
                )->Array:
        
        gt, Gt = lax.stop_gradient(grad(self.vmap_inner_product, has_aux=True)(zt, ut))
        
        return gt, Gt
    
    def loop_gt(self,
                zt:Array,
                ut:Array,
                )->Array:
        
        def step_gt(c:Tuple,
                    y:Tuple,
                    )->Tuple:
            
            z,u = y
            
            g, G = lax.stop_gradient(grad(self.loop_inner_product, has_aux=True)(z, u))
            
            return ((g,G),)*2
        
        _, (gt, Gt) = lax.scan(step_gt,
                               init=(jnp.zeros((self.T-1, self.dim), dtype=zt.dtype),
                                     jnp.zeros((self.T-1, self.dim,self.dim), dtype=zt.dtype)),
                               xs=(zt,ut),
                               )
        
        return gt, Gt
    
    def pg_gt(self,
              zt:Array,
              ut:Array,
              )->Array:
        
        gt, Gt = lax.stop_gradient(grad(self.pg_inner_product, has_aux=True)(zt, ut))
        
        return gt, Gt
    
    def create_masks(self,
                     si:Array,
                     pi:Array,
                     Gt_pg_inv_backward:Array,
                     Gt_pg_inv_forward:Array,
                     )->Array:
        
        time_idx_Gt = jnp.arange(self.T)[None, :]             # shape (1, T+1)
        mask_Gt = time_idx_Gt < pi[:, None]                          # shape (N, T+1)
        
        # Broadcast shared (T+1, d, d) arrays to (N, T+1, d, d)
        Gt_pg_inv_backward_batched = jnp.broadcast_to(Gt_pg_inv_backward, (self.N, self.T, self.dim, self.dim))
        Gt_pg_inv_forward_batched = jnp.broadcast_to(Gt_pg_inv_forward, (self.N, self.T, self.dim, self.dim))
        
        # Direction switching based on si
        full_Gt_pg_inv = jnp.where(si[:, None, None, None] == 0, Gt_pg_inv_backward_batched, Gt_pg_inv_forward_batched)  # (N, T+1, d, d)
        
        # Apply mask
        Gt_pg_inv_masked = jnp.where(mask_Gt[:, :, None, None], full_Gt_pg_inv, 0.0)  # shape (N, T+1, d, d)
        
        return Gt_pg_inv_masked
    
    def compute_A(self,
                  Git_proj_inv:Array,
                  Gi_proj_inv_sum:Array,
                  G_pg_inv_sum:Array,
                  )->Array:
        
        term1 = vmap(jnp.linalg.solve)(Gi_proj_inv_sum, G_pg_inv_sum)
        
        A = jnp.einsum('...ij,...tik,...kl->...tjl', term1
                       , Git_proj_inv, term1)
        
        return jnp.sum(A, axis=(0,1))
    
    def compute_b(self,
                  si:Array,
                  git_proj_reverse_cumsum:Array,
                  gt_pg_cumsum_reverse_forward:Array,
                  gt_pg_cumsum_added_backward:Array,
                  Git_proj_inv:Array,
                  Gi_proj_inv_sum:Array,
                  Gt_pg_inv_masked:Array,
                  G_pg_inv_sum:Array,
                  )->Array:
        
        diff = 2.*(self.z_mu-self.z_obs)
        
        term1 = (si[:,None,None] + si[:,None,None] -1)*vmap(jnp.linalg.solve)(Gi_proj_inv_sum, G_pg_inv_sum)
        term2 = jnp.sum(jnp.einsum('...ij,...j->...i', Git_proj_inv[:,:-1], git_proj_reverse_cumsum), axis=1)+diff
        term2 -= jnp.sum(jnp.einsum('...tij,...tj->...ti', Gt_pg_inv_masked[:,:-1], 
                                    si[:,None,None]*gt_pg_cumsum_reverse_forward \
                                        + (si[:,None,None]-1)*gt_pg_cumsum_added_backward), 
                                         axis=1)
        term2 = vmap(jnp.linalg.solve)(Gi_proj_inv_sum, term2)[:,None,:] - git_proj_reverse_cumsum
        term2 = jnp.concatenate((term2, vmap(jnp.linalg.solve)(Gi_proj_inv_sum, diff)[:,None,:]), axis=1)
        
        b = jnp.einsum('...ij,...tik,...tk->...tj', term1, Git_proj_inv, term2)
        
        return jnp.sum(b, axis=(0,1))
    
    def update_pi_forward(self,
                          si:Array,
                          git_proj_reverse_cumsum:Array,
                          gt_pg_cumsum_reverse_forward:Array,
                          gt_pg_cumsum_added_backward:Array,
                          Git_proj_inv:Array,
                          Gi_proj_inv_sum:Array,
                          Gt_pg_inv_masked:Array,
                          G_pg_inv_sum:Array,
                          )->Array:
        
        A = self.compute_A(Git_proj_inv,
                           Gi_proj_inv_sum,
                           G_pg_inv_sum,
                           )
        
        b = self.compute_b(si, 
                           git_proj_reverse_cumsum, 
                           gt_pg_cumsum_reverse_forward,
                           gt_pg_cumsum_added_backward,
                           Git_proj_inv,
                           Gi_proj_inv_sum,
                           Gt_pg_inv_masked,
                           G_pg_inv_sum,
                           )
        
        return jnp.linalg.solve(A,b)
    
    def update_pi_backward(self,
                           piT_forward:Array,
                           gt_pg_sum_backward:Array,
                           gt_pg_sum_forward:Array,
                           )->Array:
        
        piT_backward = -(piT_forward + gt_pg_sum_backward + gt_pg_sum_forward)
        
        return piT_backward
    
    def update_projection(self,
                          si:Array,
                          git_proj_reverse_cumsum:Array,
                          gt_pg_cumsum_added_backward:Array,
                          gt_pg_cumsum_reverse_forward:Array,
                          Git_proj_inv:Array,
                          Gi_proj_inv_sum:Array,
                          Gt_pg_inv_masked:Array,
                          piT_forward:Array,
                          )->Array:
        
        diff = 2.*(self.z_obs - self.z_mu)

        rhs = diff-jnp.sum(jnp.einsum('...tij,...tj->...ti', Git_proj_inv[:,:-1], git_proj_reverse_cumsum), axis=1) + \
            jnp.sum(jnp.einsum('...ij,...j->...i', 
                               Gt_pg_inv_masked[:,:-1], 
                               si[:,None,None]*(gt_pg_cumsum_reverse_forward + piT_forward) + \
                                   (si[:,None,None]-1)*(piT_forward + gt_pg_cumsum_added_backward),
                               ), 
                    axis=1)
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)

        muT = vmap(jnp.linalg.solve)(Gi_proj_inv_sum, rhs)[:,None,:]
        mut = jnp.concatenate((muT+git_proj_reverse_cumsum, muT), axis=1)
        
        ut_hat = -0.5*jnp.einsum('...ij,...j->...i', Git_proj_inv, mut)
        
        return ut_hat
    
    def update_pg(self,
                  piT:Array,
                  gt_pg_reverse_cumsum:Array,
                  Gt_pg_inv:Array,
                  )->Array:

        mut = jnp.vstack((piT+gt_pg_reverse_cumsum, piT))
        
        ut_hat = -0.5*jnp.einsum('tij,tj->ti', Gt_pg_inv, mut)
        
        return ut_hat
    
    def update_state(self,
                     zit_proj:Array,
                     zt_pg_backward:Array,
                     zt_pg_forward:Array,
                     si:Array,
                     pi:Array,
                     alpha:Array,
                     si_old:Array,
                     pi_old:Array,
                     si_hat:Array,
                     pi_hat:Array,
                     uit_proj_hat:Array,
                     uit_proj:Array,
                     ut_pg_hat_backward:Array,
                     ut_pg_backward:Array,
                     ut_pg_hat_forward:Array,
                     ut_pg_forward:Array,
                     )->Array:
        
        uit_proj_new = alpha*uit_proj_hat+(1.-alpha)*uit_proj
        ut_pg_new_backward = alpha*ut_pg_hat_backward+(1.-alpha)*ut_pg_backward
        ut_pg_new_forward = alpha*ut_pg_hat_forward+(1.-alpha)*ut_pg_forward
        
        zit_proj_new = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(uit_proj_new[:,:-1], axis=1)
        zt_pg_new_backward = self.z_mu+jnp.cumsum(ut_pg_new_backward, axis=0)
        zt_pg_new_forward = self.z_mu+jnp.cumsum(ut_pg_new_forward, axis=0)
        
        pi_old = jnp.where(si_old == 0, -pi_old, pi_old)
        pi_hat = jnp.where(si_hat == 0, -pi_hat, pi_hat)
        
        pi_new = jnp.round(alpha*pi_hat + (1.-alpha)*pi_old).astype(jnp.int32)
        si_new = jnp.where(pi_new < 0, 0, 1)
        pi_new = jnp.where(pi_new < 0, -pi_new, pi_new)

        return (
            zit_proj_new, 
            zt_pg_new_backward, 
            zt_pg_new_forward, 
            si_new, 
            pi_new,
            uit_proj_new, 
            ut_pg_new_backward,
            ut_pg_new_forward,
            )
    
    def update_line_search(self,
                           zit_proj:Array,
                           zt_pg_backward:Array,
                           zt_pg_forward:Array,
                           si:Array,
                           pi:Array,
                           alpha:Array,
                           si_old:Array,
                           pi_old:Array,
                           si_hat:Array,
                           pi_hat:Array,
                           uit_proj_hat:Array,
                           uit_proj:Array,
                           ut_pg_hat_backward:Array,
                           ut_pg_backward:Array,
                           ut_pg_hat_forward:Array,
                           ut_pg_forward:Array,
                           )->Array:
        
        uit_proj_new = alpha*uit_proj_hat+(1.-alpha)*uit_proj
        ut_pg_new_backward = alpha*ut_pg_hat_backward+(1.-alpha)*ut_pg_backward
        ut_pg_new_forward = alpha*ut_pg_hat_forward+(1.-alpha)*ut_pg_forward
        
        zit_proj_new = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(uit_proj_new[:,:-1], axis=1)
        zt_pg_new_backward = self.z_mu+jnp.cumsum(ut_pg_new_backward, axis=0)
        zt_pg_new_forward = self.z_mu+jnp.cumsum(ut_pg_new_forward, axis=0)
        
        pi_old = jnp.where(si_old == 0, -pi_old, pi_old)
        pi_hat = jnp.where(si_hat == 0, -pi_hat, pi_hat)
        
        pi_new = jnp.round(alpha*pi_hat + (1.-alpha)*pi_old).astype(jnp.int32)
        si_new = jnp.where(pi_new < 0, 0, 1)
        pi_new = jnp.where(pi_new < 0, -pi_new, pi_new)

        return (
            zit_proj_new, 
            zt_pg_new_backward, 
            zt_pg_new_forward, 
            si_new, 
            pi_new,
            )
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, Array, Array, Array, Array, Array, float, int],
                 )->Array:
        
        si, pi, zit_proj, zt_pg_backward, zt_pg_forward, \
            uit_proj, ut_pg_backward, ut_pg_forward, \
                Git_proj, git_proj, gt_pg_backward, gt_pg_forward, \
                    Git_proj_inv, Gt_pg_inv_backward, Gt_pg_inv_forward, \
                        error_diff, idx = carry
        
        return (error_diff>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        si, pi, zit_proj, zt_pg_backward, zt_pg_forward, \
            uit_proj, ut_pg_backward, ut_pg_forward, \
                Git_proj, git_proj, gt_pg_backward, gt_pg_forward, \
                    Git_proj_inv, Gt_pg_inv_backward, Gt_pg_inv_forward, \
                        error_diff, idx = carry
                        
        Gi_proj_inv_sum = jnp.sum(Git_proj_inv, axis=1)
        git_proj_reverse_cumsum = jnp.cumsum(git_proj[:,::-1], axis=1)[:,::-1]
        gt_pg_sum_forward = jnp.sum(gt_pg_forward, axis=0)
        gt_pg_sum_backward = jnp.sum(gt_pg_backward, axis=0)
        gt_pg_cumsum_reverse_forward = jnp.cumsum(gt_pg_forward[::-1], axis=0)[::-1]
        gt_pg_cumsum_reverse_backward = jnp.cumsum(gt_pg_backward[::-1], axis=0)[::-1]
        gt_pg_cumsum_added_backward = jnp.cumsum(gt_pg_backward, axis=0) + gt_pg_sum_forward
        
        si_hat, pi_hat = self.update_projection_points(si, 
                                                       pi, 
                                                       zt_pg_backward,
                                                       zt_pg_forward,
                                                       uit_proj, 
                                                       Git_proj, 
                                                       git_proj, 
                                                       Git_proj_inv, 
                                                       Gi_proj_inv_sum,
                                                       )
                        
        Gt_pg_inv_masked = self.create_masks(si_hat, 
                                             pi_hat,
                                             Gt_pg_inv_backward, 
                                             Gt_pg_inv_forward,
                                             )
        G_pg_inv_sum = jnp.sum(Gt_pg_inv_masked, axis=1)
        
        piT_forward = self.update_pi_forward(si_hat, 
                                             git_proj_reverse_cumsum, 
                                             gt_pg_cumsum_reverse_forward,
                                             gt_pg_cumsum_added_backward,
                                             Git_proj_inv,
                                             Gi_proj_inv_sum,
                                             Gt_pg_inv_masked,
                                             G_pg_inv_sum,
                                             )
        piT_backward = self.update_pi_backward(piT_forward, 
                                               gt_pg_sum_backward, 
                                               gt_pg_sum_forward,
                                               )
        
        ut_pg_hat_backward = self.update_pg(piT_backward, gt_pg_cumsum_reverse_backward, Gt_pg_inv_backward)
        ut_pg_hat_forward = self.update_pg(piT_forward, gt_pg_cumsum_reverse_forward, Gt_pg_inv_forward)
        
        uit_proj_hat = self.update_projection(si_hat, 
                                              git_proj_reverse_cumsum, 
                                              gt_pg_cumsum_added_backward,
                                              gt_pg_cumsum_reverse_forward,
                                              Git_proj_inv, 
                                              Gi_proj_inv_sum,
                                              Gt_pg_inv_masked,
                                              piT_forward,
                                              )
        
        zit_proj_hat = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(uit_proj_hat[:,:-1], axis=1)
        zt_pg_hat_backward = self.z_mu+jnp.cumsum(ut_pg_hat_backward, axis=0)
        zt_pg_hat_forward = self.z_mu+jnp.cumsum(ut_pg_hat_forward, axis=0)
        
        error_diff = jnp.mean(jnp.sum((zit_proj_hat - zit_proj).reshape(-1,self.dim)**2, axis=-1)) + \
            jnp.mean(jnp.sum((zt_pg_hat_backward - zt_pg_backward).reshape(-1,self.dim)**2, axis=-1)) + \
                jnp.mean(jnp.sum((zt_pg_hat_forward - zt_pg_forward).reshape(-1,self.dim)**2, axis=-1))

        tau = self.line_search((zit_proj,
                                zt_pg_backward,
                                zt_pg_forward,
                                si,
                                pi,
                                ),
                               si,
                               pi,
                               si_hat,
                               pi_hat,
                               uit_proj_hat,
                               uit_proj,
                               ut_pg_hat_backward,
                               ut_pg_backward,
                               ut_pg_hat_forward,
                               ut_pg_forward,
                               )

        zit_proj_new, zt_pg_backward_new, zt_pg_forward_new, si_new, pi_new, uit_proj_new, ut_pg_backward_new, ut_pg_forward_new, \
            = self.update_state(zit_proj, 
                                zt_pg_backward,
                                zt_pg_forward,
                                si,
                                pi,
                                tau,
                                si,
                                pi,
                                si_hat,
                                pi_hat,
                                uit_proj_hat,
                                uit_proj,
                                ut_pg_hat_backward,
                                ut_pg_backward,
                                ut_pg_hat_forward,
                                ut_pg_forward,
                                )

        git_proj, Git_proj = self.gt(zit_proj_new, uit_proj_new[:,1:])
        Git_proj_inv = jnp.concatenate((self.G0_proj_inv, jnp.linalg.inv(Git_proj)), axis=1)
        Git_proj = jnp.concatenate((self.G0_proj, Git_proj), axis=1)
        
        gt_pg_backward, Gt_pg_backward = self.pg_gt(zt_pg_backward_new[:-1], ut_pg_backward_new[1:])
        Gt_pg_inv_backward = jnp.concatenate((self.G0_pg_inv,
                                              jnp.linalg.inv(Gt_pg_backward)), axis=0)
        
        gt_pg_forward, Gt_pg_forward = self.pg_gt(zt_pg_forward_new[:-1], ut_pg_forward_new[1:])
        Gt_pg_inv_forward = jnp.concatenate((self.G0_pg_inv,
                                             jnp.linalg.inv(Gt_pg_forward)), axis=0)

        return ( 
            si_new,
            pi_new,
            zit_proj_new, 
            zt_pg_backward_new, 
            zt_pg_forward_new,
            uit_proj_new, 
            ut_pg_backward_new, 
            ut_pg_forward_new,
            Git_proj,
            git_proj, 
            gt_pg_backward,
            gt_pg_forward,
            Git_proj_inv, 
            Gt_pg_inv_backward, 
            Gt_pg_inv_forward, 
            error_diff, 
            idx+1,
            )
    
    def __call__(self,
                 z_mu:Array,
                 z_obs:Array,
                 step:str="While",
                 ):
        
        self.line_search = NaiveBacktracking(obj_fun=self.target_energy,
                                             update_fun=self.update_line_search,
                                             rho = self.rho,
                                             epsilon = self.epsilon,
                                             )
        self.z_mu = z_mu
        self.z_obs = z_obs
        self.N, self.dim = self.z_obs.shape
        
        G0_proj = vmap(self.M.G)(self.z_obs)
        self.G0_proj_inv = jnp.linalg.inv(G0_proj).reshape(self.N, 1, self.dim, self.dim)
        self.G0_proj = G0_proj.reshape(self.N, 1, self.dim, self.dim)
        
        self.G0_pg = self.M.G(z_mu)
        self.G0_pg_inv =  jnp.linalg.inv(self.G0_pg).reshape(1, self.dim, self.dim)
        
        v = self.pca(z_mu, z_obs)[0] / self.T
        
        zt_pg_backward, ut_pg_backward = self.init_pg(self.z_mu, -v)
        zt_pg_forward, ut_pg_forward = self.init_pg(self.z_mu, v)

        si, pi = self.init_projection_points(zt_pg_backward, zt_pg_forward)
        zT_proj = self.get_projection(si, pi, zt_pg_backward, zt_pg_forward)
        
        zit_proj, uit_proj = vmap(self.init_proj)(self.z_obs, zT_proj)
         
        if step == "While":
            git_proj, Git_proj = self.gt(zit_proj, uit_proj[:,1:])
            Git_proj_inv = jnp.concatenate((self.G0_proj_inv, jnp.linalg.inv(Git_proj)), axis=1)
            Git_proj = jnp.concatenate((self.G0_proj, Git_proj), axis=1)
            
            gt_pg_backward, Gt_pg_backward = self.pg_gt(zt_pg_backward[:-1], ut_pg_backward[1:])
            Gt_pg_inv_backward = jnp.concatenate((self.G0_pg_inv,
                                                  jnp.linalg.inv(Gt_pg_backward)), axis=0)
            
            gt_pg_forward, Gt_pg_forward = self.pg_gt(zt_pg_forward[:-1], ut_pg_forward[1:])
            Gt_pg_inv_forward = jnp.concatenate((self.G0_pg_inv,
                                                 jnp.linalg.inv(Gt_pg_forward)), axis=0)
            
            error_diff = self.tol + 1.
            
            si, pi, zit_proj, zt_pg_backward, zt_pg_forward, uit_proj, ut_pg_backward, ut_pg_forward, \
                Git_proj, git_proj, gt_pg_backward, gt_pg_forward, Git_proj_inv, Gt_pg_inv_backward, Gt_pg_inv_forward, \
                    error_diff, idx \
                        = lax.while_loop(self.cond_fun,
                                         self.while_step,
                                         init_val = (si,
                                                     pi,
                                                     zit_proj,
                                                     zt_pg_backward,
                                                     zt_pg_forward,
                                                     uit_proj,
                                                     ut_pg_backward,
                                                     ut_pg_forward,
                                                     Git_proj,
                                                     git_proj,
                                                     gt_pg_backward,
                                                     gt_pg_forward,
                                                     Git_proj_inv,
                                                     Gt_pg_inv_backward,
                                                     Gt_pg_inv_forward,
                                                     error_diff,
                                                     0,
                                                     ),
                                         )
            
        return (
            si, 
            pi, 
            zit_proj, 
            zt_pg_backward, 
            zt_pg_forward, 
            uit_proj, 
            ut_pg_backward, 
            ut_pg_forward, 
            error_diff, 
            idx,
            )
    
#%% Class Euclidean PCA

class PCA(ABC):
    def __init__(self,
                 )->None:
        
        self.z_mu = None
        self.z_obs = None
        self.z_centered = None
        self.U = None
        self.S = None
        self.Vh = None
        
        return
    
    def reconstruct(self,
                    pc:Array
                    )->Array:
        
        projecitons = self.z_centered @ pc
        
        reconstruct = jnp.outer(projecitons, pc)+self.z_mu
        
        return projecitons, reconstruct
    
    def __call__(self,
                 z_mu:Array,
                 z_obs:Array,
                 )->Array:
        
        self.z_mu = z_mu
        self.z_obs = self.z_obs
        self.z_centered = z_obs - z_mu
        
        self.U, self.S, self.Vh = jnp.linalg.svd(self.z_centered, full_matrices=False)
        
        return self.Vh
    
    
    
    
    
    
    
    
    
    