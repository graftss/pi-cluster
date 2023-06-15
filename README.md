# Parallel Programming Notes

This repo contains notes on building and programming a cluster comprised of multiple Raspberry Pi computers. It assumes relatively little prior knowledge, but does notably assume basic knowledge of the C programming language. Solutions to some exercises are provided. 

The document is split into five sections:

1. Assembly of the Raspberry Pi cluster, including installing necessary software on each node. The section ends by running the `xhpl` benchmark on the cluster to determine its processing power.
2. Distributed C programming on the cluster using MPI (Message Passing Interface).
3. Parallel C programming using multiple cores of a single cluster node using OpenMP (Open Multi-Processing).
4. Hybrid parallel C programming using both MPI and OpenMP.
5. An additional section on CUDA programming, which can't be completed on the Raspberry Pi cluster. 
