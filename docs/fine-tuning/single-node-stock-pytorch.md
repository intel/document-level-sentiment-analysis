# How to Run DLSA Single Node Fine-Tuning Pipeline with Stock PyTorch

## Running on CPU

### Single node

```
./fine-tuning/train_native.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/train_native.sh -h`:

```markdown
Usage: ./fine-tuning/train_native.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   ~~--bf16_ipex_ft - wether to use bf16_ipex_ft precision~~
   ~~--fp32_ipex_ft - wether to use fp32_ipex_ft precision~~
   -h | --help - displays this message
```

