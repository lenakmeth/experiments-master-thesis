# Master Thesis - Eleni Metheniti

**Linguistically inspired morphological inflection with a sequence to sequence model**

_Originally submitted and presented in September 2019 at Saarland University/DFKI Saarbrücken, published as a preprint in 2020._


## Prerequisites

- Python3
- Pytorch
- [Wikinflection Corpus](https://github.com/lenakmeth/Wikinflection-Corpus)

## Experiments 1 & 2

- make dataset for train: `python dataset.py --ĺang`
- train: `python main.py --lang` (see other parameters in `configure.py`)

Languages: names from Wikinflection (ISO 693-3)

## Experiment 3

- make small dataset for train: `python dataset_Exp3.py --ĺang`
- train: `python main_Exp3.py --lang` (see other parameters in `configure.py`)

## Cite

[Linguistically inspired morphological inflection with a sequence to sequence model](https://arxiv.org/abs/2009.02073) E. Metheniti, G. Neumann, J. van Genabith - arXiv preprint arXiv:2009.02073, 2020