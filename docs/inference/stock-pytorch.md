# How to Run DLSA Inference Pipeline with Stock PyTorch

## Support Matrix

|Categoty             |  Script |
|---|---|
|CPU Single Instance  |  single_instance.sh |
|CPU Multi Instances  |  multi_instance.sh |

> Note: Please use the fine-tuned model for correct accuracy. Just change the `MODEL_NAME_OR_PATH` in the script before you running. By default, the `MODEL_NAME_OR_PATH` is `bert-large-uncased` which is downloaded from the Hugging Face website.

## Running on CPU

### Single instance

```
./inference/single_instance.sh
```

By default, it will launch 1 instance to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/single_instance.sh -h`:

```markdown
Usage: ./inference/single_instance.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   ~~--ipex_fp32 - wether to use ipex_fp32 precision~~
   ~~--ipex_bf16 - wether to use ipex_bf16 precision~~
   ~~--int8 - wether to use int8 precision~~
   ~~--int8_bf16 - wether to use int8_bf16 precision~~
   -h | --help - displays this message
```



### Multi-instance

```
./inference/multi_instance.sh
```

By default, it will launch 2 instances (1 instance/socket) to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/multi_instance.sh -h`

```markdown
Usage: ./inference/multi_instance.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -n | --num_of_ins_per_socket - number of instance per socket
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   ~~--ipex_fp32 - wether to use ipex_fp32 precision~~
   ~~--ipex_bf16 - wether to use ipex_bf16 precision~~
   ~~--int8 - wether to use int8 precision~~
   ~~--int8_bf16 - wether to use int8_bf16 precision~~
   -h | --help - displays this message
```

