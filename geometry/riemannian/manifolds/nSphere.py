#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from .manifold import RiemannianManifold
from .nEllipsoid import nEllipsoid

#%% Code

class nSphere(nEllipsoid):
    def __init__(self,
                 dim:int=2,
                 coordinates="stereographic",
                 )->None:
        super().__init__(dim=dim, params=jnp.ones(dim+1, dtype=jnp.float32), coordinates=coordinates)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def Exp(self,
            x:Array,
            v:Array,
            t:float=1.0,
            )->Array:
        
        norm = jnp.linalg.norm(v)
        
        return lax.cond(norm < 1e-6,
                        lambda *_: x,
                        lambda *_: (jnp.cos(norm*t)*x+jnp.sin(norm*t)*v/norm)*self.params,
                        )
    
    def Geodesic(self,
                 z0:Array,
                 zN:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,99, endpoint=False)[1:].reshape(-1,1)
        else:
            t_grid = t_grid.reshape(-1,1)
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        
        z0_norm = jnp.linalg.norm(z0)
        zN_norm = jnp.linalg.norm(zN)
        dot_product = jnp.dot(z0, zN)
        theta = jnp.arccos(dot_product/(z0_norm*zN_norm))
        
        sin_theta = jnp.sin(theta)
        
        curve = lax.cond(jnp.abs(sin_theta) < 1e-6,
                         lambda *_: z0+jnp.zeros((len(t_grid), *shape), dtype=z0.dtype),
                         lambda *_: ((z0*jnp.sin((1.-t_grid)*theta) + zN*jnp.sin(t_grid*theta))/sin_theta),
                         )
        
        curve = jnp.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)
    
    
    