# NLU Dataset Diagnostics

This repository contains data and scripts to reproduce the results from our paper:

Aarne Talman, Marianna Apidianaki, Stergios Chatzikyriakidis, Jörg Tiedemann. 2022. How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets.

## Reproduce our results
Install the dependencies by running:
```bash
pip install -r requirements.txt
```

Run the experiments using the following command:
```bash
bash run_experiment.sh
```

The Python script `run_corrupt_glue.py` is a modified version of the
`run_glue.py` script by Huggingface available in their [Text classification examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification).

