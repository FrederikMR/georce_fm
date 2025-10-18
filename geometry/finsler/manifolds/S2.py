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

from .manifold import FinslerManifold

#%% Code

class S2(FinslerManifold):
    def __init__(self,
                 )->None:
        self.dim = 2
        self.emb_dim = 3
        super().__init__(F=self.F, f=self.f, invf=self.invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def F(self,
          z:Array,
          tv:Array,
          )->Array:
        
        u,v = z
        x,y = tv
         
        a = 1. + (u**2) + (v**2)
        b = 1. + (u**2) + (v**2)
        c1 = 0.75
        c2 = 0
        theta = 0.0
        
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
    
        term1 = x * sin_theta + y * cos_theta
        term2 = x * cos_theta - y * sin_theta
    
        numerator = (
            -a**2 * c2 * term1
            - b**2 * c1 * term2
            + jnp.sqrt(
                a**4 * b**2 * term1**2
                + a**2 * b**4 * term2**2
                - a**2 * b**2 * c1**2 * term1**2
                + 2 * a**2 * b**2 * c1 * c2 * term2 * term1
                - a**2 * b**2 * c2**2 * term2**2
            )
        )
    
        denominator = a**2 * b**2 - a**2 * c2**2 - b**2 * c1**2
    
        return numerator / denominator
    
    def f(self, 
          z:Array,
          )->Array:
        
        u = z[..., 0]
        v = z[..., 1]
        d = 1 + u**2 + v**2
    
        r1 = (2 * u) / d
        r2 = (2 * v) / d
        r3 = (-1 + u**2 + v**2) / d
    
        return 0.5 * jnp.stack([r1, r2, r3], axis=-1)

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