"""Vendored markov_clustering 0.0.6.dev API used by the MCL baseline."""

from .mcl import (
    add_self_loops,
    converged,
    expand,
    get_clusters,
    inflate,
    iterate,
    normalize,
    prune,
    run_mcl,
)

__version__ = "0.0.6.dev"
__author__ = "Guy Allard"

__all__ = [
    "add_self_loops",
    "converged",
    "expand",
    "get_clusters",
    "inflate",
    "iterate",
    "normalize",
    "prune",
    "run_mcl",
]
