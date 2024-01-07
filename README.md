# [Model] Prime


Updated on 2023.10.20


## Introduction

This repository provides the official implementation of Prime (Protein language model for Intelligent Masked pretraining and Environment (temperature) prediction).

Key feature:
- Zero-shot mutant effect prediction.

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/ai4protein/Prime) 

## Details

### What is Prime?
Prime, a novel protein language model, has been developed for predicting the Optimal Growth Temperature (OGT) and enabling zero-shot prediction of protein thermostability and activity. This novel approach leverages temperature-guided language modeling.
![logo](./model.png)


## Use of PRIME

**Main Requirements**  
biopython==1.81
torch==2.0.1

**Installation**
```bash
pip install -r requirements.txt
```

**Download checkpoints of PRIME**

[Our official files website](https://lianglab.sjtu.edu.cn/files/Prime-2023/)

For example
```shell
wget -O checkpoints/prime_base.pt https://lianglab.sjtu.edu.cn/files/Prime-2023/Base_model/prime_base.pt
```

**Predicting Mutant Effect**
```shell
python predict.py --model_path checkpoints/prime_base.pt \
--fasta ./tm_data/fasta/O25949-7.2.fasta \
--mutant tm_data/mutant/O25949-7.2.csv \
--save O25949-7.2.prime_base.csv
```

## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers) and [esm](https://github.com/facebookresearch/esm).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023,
      title={Engineering Enhanced Stability and Activity in Proteins through a Novel Temperature-Guided Language Modeling.}, 
      author={Pan Tan and Mingchen Li and Liang Zhang and Zhiqiang Hu and Liang Hong},
      year={2023},
      eprint={2304.03780},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```