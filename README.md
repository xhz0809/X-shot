# X-Shot: A Unified System to Handle Frequent, Few-shot and Zero-shot Learning Simultaneously in Classification

This repository contains codes and scripts relevant to the dataset introduced in this [paper](#link)


<!-- ## Requirements -->

## Code Structure
 - `src/`: contains the scripts used for the experiments.
 - `data/`: contains the original and transformed data used in the paper. 

## Fine-tune Binbin 
```
python src/Binbin/mian.py  \
  --model_name_or_path t5-3B \
  --train_file data/RAMS/T2/RAMS_instruction_train.json  \
  --validation_file  data/RAMS/T2/RAMS_instruction_dev.json  \
  --max_length 512  \
  --per_device_train_batch_size 16  \
  --per_device_eval_batch_size 32  \
  --learning_rate 5e-5 \ 
  --num_train_epochs 10\
  --checkpointing_steps  epoch \
  --seed 42   \
  --mix_precision  bf16 \
  --data_name RAMS  \
  --output_dir checkpoints/RAMS/T5/5e5 \
```
 - The fine-tuning process can be optional. The model can be directly applied to any dataset as a zero-shot approach
 - To fine-tune the model on a new dataset, make sure you modify the file path
 - Note that the model can be any existing architecture. We include the three most common transformer architectures: encoder-only (RoBERTa), encoder-decoder (T5), and decoder-only (GPT-Neo)
<!-- ## Dataset -->


<!-- ## Model -->

## Citation 
Please cite the following work if you want to refer to this work: 
```

```

## Contact
Hanzi Xu(hanzi.xu@temple.edu)
