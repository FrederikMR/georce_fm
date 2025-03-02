#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% LorentFinslerManifold

class LorentzFinslerManifold(ABC):
    def __init__(self,
                 F:Callable[[Array, Array, Array], Array],
                 G:Callable[[Array, Array, Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
                 )->None:
        
        self.F = F
        self.f = f
        self.inv = invf
        
        if  not (G is None):
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Lorentz Finsler Manifold base object"
    
    def G(self, t:Array, z:Array, v:Array)->Array:
        
        return 0.5*jacfwd(lambda v1: grad(lambda v2: self.F(t,z,v2)**2)(v1))(v)
    
    def g(self, t:Array, z:Array, v:Array)->Array:
        
        G = self.G(t,z,v)
        
        return jnp.einsum('i,ij,j->', v, G, v)
    
    def Ginv(self, t:Array,z:Array, v:Array)->Array:
        
        return jnp.linalg.pinv(self.G(t,z,v))
    
    def Dg(self, t:Array, z:Array, v:Array)->Array:
        
        return jacfwd(self.G, argnums=1)(t,z,v)
    
    def geodesic_equation(self, 
                          t:Array,
                          z:Array, 
                          v:Array
                          )->Array:
        
        g = self.G(t,z,v)
        Dg = self.Dg(t,z,v)
        
        rhs = jnp.einsum('ikj,i,j->k', Dg, v, v)-0.5*jnp.einsum('ijk,i,j->k', Dg, v, v)
        rhs = jnp.linalg.solve(g, rhs)
        
        dx1t = v
        dx2t = -rhs
        
        return jnp.vstack((dx1t,dx2t))        
    
    def energy(self, 
               t:Array,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        s = jnp.linspace(0.,1.,T,endpoint=False)

        integrand = vmap(lambda t,g,dg,s: self.F(t,g,dg,s)**2)(t[:-1],gamma[:-1], dgamma, s)

        return jnp.trapz(integrand, dx=dt)
    
    def length(self,
               t:Array,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T

        integrand = vmap(lambda t,g,dg: self.F(t,g,dg))(t[:-1],gamma[:-1],dgamma)
            
        return jnp.trapz(integrand, dx=dt)
    
    def length_frechet(self, 
                       t0:Array,
                       ts:Array,
                       zs:Array,
                       z_obs:Array,
                       z_mu:Array,
                       )->Array:
        
        def step_length(length:Array,
                        y:Tuple,
                        )->Tuple:
            
            z0, z_path, t_path = y
            
            length += self.path_length_frechet(z0, z_path, z_mu, t_path, t0)**2
            
            return (length,)*2
        
        length, _ = lax.scan(step_length,
                             init=0.0,
                             xs=(z_obs, zs, ts),
                             )

        return length
    
    def path_length_frechet(self, 
                            zT:Array,
                            zs:Array,
                            mu:Array,
                            ts:Array,
                            t0:Array,
                            )->Array:
        
        term1 = zs[0]-mu
        val1 = self.F(t0, mu, term1)
        
        term2 = zs[1:]-zs[:-1]
        val2 = vmap(lambda t,z,v: self.F(t,z,v))(ts[:-1], zs[:-1],term2)
        
        term3 = zT-zs[-1]
        val3 = self.F(ts[-1], zs[-1],term3)
        
        return val1+jnp.sum(val2)+val3
    
    def time_fun(self,
                 t0:Array,
                 zt:Array,
                 )->Array:
        
        t = self.time_integral(t0,
                               zt[:-1],
                               zt[-1],
                               )[:-1]
        
        return t
    
    def time_integral(self,
                      t0:Array,
                      zt:Array,
                      zT:Array,
                      )->Array:
        
        def time_update(t:Array,
                        step:Tuple[Array,Array],
                        )->Array:
            
            z, dz = step
            
            t += self.F(t, z, dz)
            
            return (t,)*2

        dz = jnp.vstack((zt[1:]-zt[:-1], zT-zt[-1]))
        _, t = lax.scan(time_update,
                        init=t0,
                        xs = (zt, dz),
                        )
        
        return t
    
    def indicatrix(self,
                   t:Array,
                   z:Array,
                   s:Array,
                   N_points:int=100,
                   *args,
                   )->Array:
        
        theta = jnp.linspace(0.,2*jnp.pi,N_points)
        u = jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
        
        norm = vmap(self.F, in_axes=(None, None, 0, None))(t,z,u,s)
        
        return jnp.einsum('ij,i->ij', u, 1./norm)
    
    def indicatrix_opt(self,
                       t:Array,
                       z:Array,
                       grid:Array=None,
                       eps:float=1e-4,
                       )->Array:
        
        def minimizer(u0:Array,
                      reverse:bool=False
                      )->Array:
            
            if reverse:
                u = jminimize(obj_fun, 
                              x0=jnp.ones(1, dtype=z.dtype), 
                              args=(True, u0), 
                              method="BFGS", tol=1e-4, 
                              options={'maxiter':100}).x
                u = jnp.vstack((jnp.hstack((u, u0)),
                                jnp.hstack((-u, u0)))
                               )
            else:
                u = jminimize(obj_fun, 
                              x0=jnp.ones(1, dtype=z.dtype), 
                              args=(u0, False), 
                              method="BFGS", tol=1e-4, 
                              options={'maxiter':100}).x
                
                u = jnp.vstack((jnp.hstack((u0, u)),
                                jnp.hstack((u0, -u)))
                               )
            
            return u
        
        def obj_fun(ui:Array,
                    u0:Array,
                    reverse:bool=False,
                    )->Array:
            
            if reverse:
                u = jnp.hstack((ui,u0))
            else:
                u = jnp.hstack((u0,ui))

            return (self.F(t,z,u)-1.0)**2
        
        if grid is None:
            grid = jnp.linspace(-5.0,5.0,10)

        u11 = vmap(lambda u0: jnp.hstack((u0, jminimize(obj_fun, 
                                                       x0=jnp.ones(1, dtype=z.dtype), 
                                                       args=(u0, False), 
                                                       method="BFGS", tol=eps, 
                                                       options={'maxiter':100}).x)))(grid)
        u12 = vmap(lambda u0: jnp.hstack((u0, jminimize(obj_fun, 
                                                       x0=-jnp.ones(1, dtype=z.dtype), 
                                                       args=(u0, False), 
                                                       method="BFGS", tol=eps, 
                                                       options={'maxiter':100}).x)))(grid)
        u1 = jnp.concatenate((u11, u12), axis=0)
        
        u21 = vmap(lambda u0: jnp.hstack((jminimize(obj_fun, 
                                                   x0=-jnp.ones(1, dtype=z.dtype),
                                                   args=(u0, True),
                                                   method="BFGS", tol=eps,
                                                   options={'maxiter':100}).x,
                                         u0)))(grid)
        u22 = vmap(lambda u0: jnp.hstack((jminimize(obj_fun, 
                                                   x0=-jnp.ones(1, dtype=z.dtype),
                                                   args=(u0, True),
                                                   method="BFGS", tol=eps,
                                                   options={'maxiter':100}).x,
                                         u0)))(grid)
        u2 = jnp.concatenate((u21, u22), axis=0)
        
        #print(jnp.mean(u1[:,1]))
        #print(jnp.mean(u2[:,0]))
        
        #u1_reverse = jnp.vstack((u1[:,0], jnp.mean(u1[:,1])-u1[:,1])).T
        #u2_reverse = jnp.vstack((jnp.mean(u2[:,0])-u2[:,0], u2[:,1])).T
        
        #u1_reverse = jnp.vstack((-u1[:,0], u1[:,1])).T
        #u = jnp.concatenate((u1,u1_reverse), axis=0)
        u = jnp.concatenate((u1, u2), axis=0)
        #u = jnp.concatenate((u1,u2,u1_reverse,u2_reverse), axis=0)
        length = vmap(self.F, in_axes=(None, None, 0))(t,z, u)
        u = u[(length-1.0)**2 < eps]
        #u = jnp.sort(u, axis=0)
        
        theta = vmap(jnp.arctan2)(u[:,0],u[:,1])
        
        return u[theta.argsort()]
    