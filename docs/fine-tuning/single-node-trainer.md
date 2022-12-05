# How to Run DLSA Single Node Fine-Tuning with Trainer(FP32, BF16)

## Running on CPU

### Single node

```
./fine-tuning/train_trainer.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/train_native.sh -h`:

```markdown
Usage: ./fine-tuning/train_trainer.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --bf16 - whether using hf bf16 inference
   --use_ipex - whether using ipex
   -h | --help - displays this message
```



