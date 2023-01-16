# How to Run DLSA Fine-Tuning with IPEX(FP32, BF16)

## Support Matrix

| Categoty             | Script          |
| -------------------  | --------------- |
| IPEX Single Instance | ft_ipex.sh      |
| IPEX Multi Instances | ft_ipex_ccl.sh  |

## Single Instance Fine-Tuning

```
./fine-tuning/ft_ipex.sh 
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/ft_ipex.sh -h`:

```markdown
Usage: ./fine-tuning/ft_ipex.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --dtype_ft - data type used for fine-tuning
   --train_epoch - train epoch
   -h | --help - displays this message
```
## Multi Instances Fine-Tuning


### Running single instance

```
bash fine-tuning/run_dist.sh -np 1 -ppn 1 bash fine-tuning/ft_ipex_ccl.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

### Running multi instances

```
bash fine-tuning/run_dist.sh -np 2 -ppn 2 bash fine-tuning/ft_ipex_ccl.sh
```

By default, it will launch 2 instances on single node to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

> Note:
>
> np: num process, means how many processes you will run on a cluster
>
> ppn: process per node, means how many processes you will run on 1 worker node.
>
> For example, if I want to run on 2 nodes, each node runs with 1 process, use the config `-np 2 -ppn 1`
>
> if I want to run on 4 nodes, each node runs with 2 processes, use the config `-np 8 -ppn 2`
>
> You can also use `-l $log_name` after `run_dist.sh` to set the log name.
