# How to Run DLSA Inference Pipeline with HF Transformers(FP32, BF16)

## Support Matrix

|Categoty             |  Script |
|---|---|
|CPU Single Instance  |  inf_trainer_single.sh |
|CPU Multi Instances  |  inf_trainer_multi.sh |

> Note: Please use the fine-tuned model for correct accuracy. Just change the `MODEL_NAME_OR_PATH` in the script before you running. By default, the `MODEL_NAME_OR_PATH` is `bert-large-uncased` which is downloaded from the Hugging Face website.

## Running on CPU

### Single instance

```
./inference/inf_trainer_single.sh
```

By default, it will launch 1 instance to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/inf_trainer_single.sh -h`:

```markdown
Usage: ./inference/inf_trainer_single.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --dtype_inf - data type used for inference
   -h | --help - displays this message
```



### Multi-instance

```
./inference/inf_trainer_multi.sh
```

By default, it will launch 2 instances (1 instance/socket) to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/inf_trainer_multi.sh -h`

```markdown
Usage: ./inference/inf_trainer_multi.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET
   -n | --num_of_ins_per_socket - number of instance per socket
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --dtype_inf - data type used for inference
   -h | --help - displays this message
```
