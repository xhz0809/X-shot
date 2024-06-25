# X-Shot: A Unified System to Handle Frequent, Few-shot and Zero-shot Learning Simultaneously in Classification

This repository contains codes and scripts relevant to the dataset introduced in this [paper](https://arxiv.org/abs/2403.03863).


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
@article{DBLP:journals/corr/abs-2403-03863,
  author       = {Hanzi Xu and
                  Muhao Chen and
                  Lifu Huang and
                  Slobodan Vucetic and
                  Wenpeng Yin},
  title        = {X-Shot: {A} Unified System to Handle Frequent, Few-shot and Zero-shot
                  Learning Simultaneously in Classification},
  journal      = {CoRR},
  volume       = {abs/2403.03863},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.03863},
  doi          = {10.48550/ARXIV.2403.03863},
  eprinttype    = {arXiv},
  eprint       = {2403.03863},
  timestamp    = {Tue, 07 May 2024 20:16:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-03863.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact
Hanzi Xu(hanzi.xu@temple.edu)
