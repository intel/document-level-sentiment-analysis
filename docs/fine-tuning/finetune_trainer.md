# How to Run DLSA Single Node Fine-Tuning with HF Trainer(FP32, BF16)

## Single instance Fine-Tuning

```
./fine-tuning/ft_trainer.sh
```

By default, it will launch 1 instance to run fine-tuning with SST-2 dataset and FP32 precision. You can change the configurations in the file or pass parameters when running the script.

Below is the help message by using the command `./fine-tuning/ft_trainer.sh -h`:

```markdown
Usage: ./fine-tuning/ft_trainer.sh [OPTIONS]
OPTION includes:
   -l | --log_name - the log name of this round
   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET
   -b | --batch_size - batch size per instance
   -s | --sequence_len - max sequence length
   --dtype_ft - data type used for fine-tuning
   -h | --help - displays this message
```
