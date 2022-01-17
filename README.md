# NLU Dataset Diagnostics

This repository contains data and scripts to reproduce the results from our paper:

Aarne Talman, Marianna Apidianaki, Stergios Chatzikyriakidis, Jörg Tiedemann. 2022. [How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets](https://arxiv.org/abs/2201.04467).


*A central question in natural language understanding (NLU) research is whether high performance demonstrates the models' strong reasoning capabilities. We present an extensive series of controlled experiments where pre-trained language models are exposed to data that have undergone specific corruption transformations. The transformations involve removing instances of specific word classes and often lead to non-sensical sentences. Our results show that performance remains high for most GLUE tasks when the models are fine-tuned or tested on corrupted data, suggesting that the models leverage other cues for prediction even in non-sensical contexts. Our proposed data transformations can be used as a diagnostic tool for assessing the extent to which a specific dataset constitutes a proper testbed for evaluating models' language understanding capabilities.*

## Reproduce our results
Install the dependencies by running:
```bash
pip install -r requirements.txt
```

Run the experiments using the following command:
```bash
bash run_experiment.sh
```

`run_experiment.sh` starts a fine-tuning job for each configuration and only works in an environment where you have access to a lot of GPU instances managed with an orchestration system like SLURM. To run a single configuration, you can modify `train.sh` and tun it:

```bash
bash train.sh
```

The Python script `run_corrupt_glue.py` is a modified version of the
`run_glue.py` script by Huggingface available in their [Text classification examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification).

## Cite our paper

```
@misc{talman_et_al2022,
      title={How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets}, 
      author={Aarne Talman and Marianna Apidianaki and Stergios Chatzikyriakidis and J\"org Tiedemann},
      year={2022},
      eprint={2201.04467},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

