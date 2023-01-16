# How to Run DLSA Inference Pipeline with IPEX(FP32, BF16, INT8)

## Support Matrix

| Categoty            | Script             |
| ------------------- | ------------------ |
| IPEX Single Instance | inf_ipex_single.sh |
| IPEX Multi Instances | inf_ipex_multi.sh  |

> Note: Please use the fine-tuned model for correct accuracy. Just change the `MODEL_NAME_OR_PATH` in the script before you running. By default, the `MODEL_NAME_OR_PATH` is `bert-large-uncased` which is downloaded from the Hugging Face website.

## Running on CPU

> Note: For int8 inference, you need to quantize the model firstly. Please see the details in this link:  https://github.com/IntelAI/models/tree/master/quickstart/language_modeling/pytorch/bert_large/inference/cpu

### Single instance

```
./inference/inf_ipex_single.sh
```

By default, it will launch 1 instance to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/inf_ipex_single.sh -h`:

```markdown
Usage: ./inference/inf_ipex_single.sh [OPTIONS]
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
./inference/inf_ipex_multi.sh
```

By default, it will launch 2 instances (1 instance/socket) to run inference with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./inference/inf_ipex_multi.sh -h`

```markdown
Usage: ./inference/inf_ipex_multi.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET
   -n | --num_of_ins_per_socket - number of instance per socket
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --dtype_inf - data type used for inference
   -h | --help - displays this message
```
