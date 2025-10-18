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

#%% Code

class S2(RiemannianManifold):
    def __init__(self,
                 )->None:
        self.dim = 2
        self.emb_dim = 3
        super().__init__(f=self.f, invf=self.invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def f(self, 
          z:Array,
          )->Array:
        
        u,v = z
        
        D = 1 + u**2 + v**2
        x = u / D
        y = v / D
        z = (-1 + u**2 + v**2) / (2 * D)
        
        return jnp.array([x, y, z])

    def invf(self, 
             x_embedded:Array,
             )->Array:
        
        x = x_embedded[..., 0]
        y = x_embedded[..., 1]
        z = x_embedded[..., 2]
    
        denom = 1.0 - z
        u = x / denom
        v = y / denom
    
        return jnp.stack([u, v], axis=-1)