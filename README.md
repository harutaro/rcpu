# RCPU
### [ICLR2026] RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models
This is a reference code for RCPU.
[Arxiv link](https://arxiv.org/abs/2510.07782)

## Requirements
We conducted experiments in Nvidia A100 GPU with CUDA12.6.

    conda create -n rcpu python=3.10.16
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    pip install transformers==4.57.1 datasets==4.6.0

## Usage
The command like below will start pruning and ppl evaluation.

    CUDA_VISIBLE_DEVICES=0 python main.py --method rcpu --unstr --nsamples 128 --pruning_ratio 0.1

Note: Minor numerical differences may occur depending on library versions and hardware configurations.

## Bibtex

~~~
@inproceedings{haruta2026rcpu,
  title={RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models},
  author={Haruta, Shuichiro and Matsumoto, Kazunori and Li, Zhi and Wang, Yanan and Kurokawa, Mori},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
~~~

## Note

- In our experiments, benchmark evaluations are done by [LM-EVALUATION-HARNESS](https://github.com/EleutherAI/lm-evaluation-harness).
- We made this project based on [FLAP](https://github.com/CASIA-LMC-Lab/FLAP).

Thanks!
