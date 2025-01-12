#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.random as jrandom
import jax.numpy as jnp
from jax import jit, lax

import haiku as hk

from geometry.riemannian.manifolds import nSphere, nEllipsoid, nEuclidean, \
    nParaboloid, HyperbolicParaboloid, SPDN, H2, Cylinder, Landmarks, T2, \
        LatentSpaceManifold, FisherRaoGeometry
    
from vae.model_loader import mnist_generator, svhn_generator, celeba_generator, load_model

from vae.models import mnist_encoder
from vae.models import mnist_decoder

from vae.models import svhn_encoder
from vae.models import svhn_decoder

from vae.models import celeba_encoder
from vae.models import celeba_decoder

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", 
                  dim:int = 2,
                  N_data:int=100,
                  data_path:str = 'data/',
                  ):
    
    load_path = ''.join((data_path, f'{manifold}/'))
    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
    if manifold == "SPDN":
        M = SPDN(N=dim)
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
    elif manifold == "Sphere":
        M = nSphere(dim=dim)
    elif manifold == "Ellipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
    elif manifold == "H2":
        M = H2()
    elif manifold == "Cylinder":
        M = Cylinder()
    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
    elif manifold == "Landmarks":
        M = Landmarks(N=dim, m=2, k_alpha=0.1)
    elif manifold == "Gaussian":
        M = FisherRaoGeometry(distribution='Gaussian')
    elif manifold == "Frechet":
        M = FisherRaoGeometry(distribution='Frechet')
    elif manifold == "Cauchy":
        M = FisherRaoGeometry(distribution='Cauchy')
    elif manifold == "Pareto":
        M = FisherRaoGeometry(distribution='Pareto')
    elif manifold == "celeba":
        celeba_state = load_model(''.join(('models/', f'celeba_{dim}/')))
       
        @hk.transform
        def celeba_tencoder(x):
       
            encoder = celeba_encoder(latent_dim=32)
       
            return encoder(x)[0]
       
        @hk.transform
        def celeba_tdecoder(x):
       
            decoder = celeba_decoder()
       
            return decoder(x)
       
        celeba_encoder_fun = jit(lambda x: celeba_tencoder.apply(lax.stop_gradient(celeba_state.params),
                                                                 None,
                                                                 x.reshape(-1,64,64,3)
                                                                 )[0].reshape(-1,dim).squeeze())
        celeba_decoder_fun = jit(lambda x: celeba_tdecoder.apply(lax.stop_gradient(celeba_state.params),
                                                                 None,
                                                                 x.reshape(-1,dim)
                                                                 ).reshape(-1,64*64*3).squeeze())
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=64*64*3,
                                encoder=celeba_encoder_fun,
                                decoder=celeba_decoder_fun,
                                )
   
    elif manifold == "svhn":
        svhn_state = load_model(''.join(('models/', f'svhn_{dim}/')))
       
        @hk.transform
        def svhn_tencoder(x):
       
            encoder = svhn_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def svhn_tdecoder(x):
       
            decoder = svhn_decoder()
       
            return decoder(x)
       
        svhn_encoder_fun = jit(lambda x: svhn_tencoder.apply(lax.stop_gradient(svhn_state.params),
                                                             None,
                                                             x.reshape(-1,32,32,3)
                                                             )[0].reshape(-1,dim).squeeze())
        svhn_decoder_fun = jit(lambda x: svhn_tdecoder.apply(lax.stop_gradient(svhn_state.params),
                                                             None,
                                                             x.reshape(-1,dim)
                                                             ).reshape(-1,32*32*3).squeeze())
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=32*32*3,
                                encoder=svhn_encoder_fun,
                                decoder=svhn_decoder_fun,
                                )
        
    elif manifold == "mnist":
        mnist_state = load_model(''.join(('models/', f'mnist_{dim}/')))
       
        @hk.transform
        def mnist_tencoder(x):
       
            encoder = mnist_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def mnist_tdecoder(x):
       
            decoder = mnist_decoder()
       
            return decoder(x)
       
        mnist_encoder_fun = lambda x: mnist_tencoder.apply(mnist_state.params,
                                                           None,
                                                           x.reshape(-1,28,28,1)
                                                           )[0].reshape(-1,dim).squeeze()
        mnist_decoder_fun = lambda x: mnist_tdecoder.apply(mnist_state.params,
                                                           None,
                                                           x.reshape(-1,dim)
                                                           ).reshape(-1,28*28).squeeze()
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=28*28,
                                encoder=mnist_encoder_fun,
                                decoder=mnist_decoder_fun,
                                )
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    z_obs = jnp.load(''.join((load_path, f'z_obs_{dim}_{N_data}.npy')))
        
    return z_obs, M