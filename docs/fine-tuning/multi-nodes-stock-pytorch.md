# How to Run DLSA Multi Node Fine-Tuning with Stock PyTorch(FP32)

## Install MPI library:

Install MPI from [here]( https://anaconda.org/intel/impi_rt )


MPI is included in the Intel OneAPI Toolkit. It's recommended to use the package manager to install.

>  Note: This step should be operated on all the work nodes

## To run:

```
source /opt/intel/oneapi/mpi/latest/env/vars.sh
cd profiling-transformers
```

> Note:
>
> np: num process, means how many processes you will run on a cluster
>
> ppn: process per node, means how many processes you will run on 1 worker node.
>
> For example, if I want to run on 2 nodes, each node runs with 1 process, use the config `-np 2 -ppn 1`
>
> if I want to run on 4 nodes, each node runs with 2 processes, use the config `-np 8 -ppn 2`

### Running single process in single node

```
bash fine-tuning/run_dist.sh -np 1 -ppn 1 bash fine-tuning/run_ipex_native.sh
```

### Running multi-node fine-tuning

> You need to create the `hostfile` which contains all nodes you want to run on and set password-free login.

```
bash fine-tuning/run_dist.sh -np 2 -ppn 1 -f hostfile bash fine-tuning/run_ipex_native.sh
```



