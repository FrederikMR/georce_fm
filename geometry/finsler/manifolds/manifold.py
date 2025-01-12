#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% Riemannian Manifold

class FinslerManifold(ABC):
    def __init__(self,
                 F:Callable[[Array, Array], Array],
                 G:Callable[[Array, Array], Array]=None,
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
        
        return "Finsler Manifold base object"
    
    def g(self, z:Array, v:Array)->Array:
        
        G = self.G(z,v)
        
        return jnp.einsum('i,ij,j->', v, G, v)
    
    def G(self, z:Array, v:Array)->Array:
        
        return 0.5*jacfwd(lambda v1: grad(lambda v2: self.F(z,v2)**2)(v1))(v)
    
    def Ginv(self, z:Array, v:Array)->Array:
        
        return jnp.linalg.inv(self.G(z,v))
    
    def Dgv(self, z:Array, v:Array)->Array:
        
        return jacfwd(self.gv)(z,v)
    
    def geodesic_equation(self, 
                          z:Array, 
                          v:Array
                          )->Array:
        
        gv = self.gv(z,v)
        Dgv = self.Dgv(z,v)
        
        rhs = jnp.einsum('ikj,i,j->k', Dgv, v, v)-0.5*jnp.einsum('ijk,i,j->k', Dgv, v, v)
        rhs = jnp.linalg.solve(gv, rhs)
        
        dx1t = v
        dx2t = -rhs
        
        return jnp.hstack((dx1t,dx2t))        
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T

        integrand = vmap(lambda g,dg: self.F(g,dg)**2)(gamma[:-1], dgamma)

        return jnp.trapz(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T

        integrand = vmap(lambda g,dg: self.F(g,dg))(gamma[:-1],dgamma)
            
        return jnp.trapz(integrand, dx=dt)
    
    def length_frechet(self, 
                       zt:Array,
                       z_obs:Array,
                       z_mu:Array,
                       )->Array:

        path_length = vmap(self.path_length_frechet, in_axes=(0,0,None))(z_obs, zt, z_mu)
        
        return jnp.sum(path_length**2)
    
    def path_length_frechet(self, 
                            zT:Array,
                            zt:Array,
                            mu:Array,
                            )->Array:
        
        term1 = zt[0]-mu
        val1 = self.F(mu, term1)
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda z,v: self.F(z,v))(zt[:-1],term2)
        
        term3 = zT-zt[-1]
        val3 = self.F(zt[-1],term3)
        
        return val1+jnp.sum(val2)+val3
    