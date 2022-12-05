# How to Run DLSA Inference Pipeline with HF Transformers(FP32, BF16)

## Support Matrix

|Categoty             |  Script |
|---|---|
|CPU Single Instance  |  cpu_single_instance.sh |
|CPU Multi Instances  |  cpu_multi_instance.sh |

> Note: Please use the fine-tuned model for correct accuracy. Just change the `MODEL_NAME_OR_PATH` in the script before you running. By default, the `MODEL_NAME_OR_PATH` is `bert-large-uncased` which is downloaded from the Hugging Face website.

## Running on CPU

### Single instance

```
./inference/cpu_single_instance.sh
```

By default, it will launch 1 instance to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/cpu_single_instance.sh -h`:

```markdown
Usage: ./inference/cpu_single_instance.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --bf16 - whether using hf bf16 inference
   --use_ipex - whether using ipex
   -h | --help - displays this message
```



### Multi-instance

```
./inference/cpu_multi_instance.sh
```

By default, it will launch 2 instances (1 instance/socket) to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/cpu_multi_instance.sh -h`

```markdown
Usage: ./inference/cpu_multi_instance.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -n | --num_of_ins_per_socket - number of instance per socket
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --bf16 - whether using hf bf16 inference
   --use_ipex - whether using ipex
   -h | --help - displays this message
```
