# Transformer Playground

To understand Transformer Architectures through practice—from implementing one from scratch to adapting pre-trained models (BERT, GPT). 

| Notebook | Concepts | English | Russian |
|----------|----------|---------|---------|
| 1. Tokenization and Encoding | BPE, HuggingFace Tokenizers, Collator | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/qmarva/bpe-tokenization) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) |
| 2. Transformer Architecture | Positional Encoding, Attention(KQV, Multi-Head), Encoder-Decoder | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/qmarva/building-transformer) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) |
| 3. Functions, Metrics, Tools | BLEU, ROGUE, METEOR, WandB | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) |
| 4. Transformer Training | LR Scheduler, Xavier Initialization, LabelSmoothing, Hyperparameter Tuning | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) |
| 5. Complete Transformer | End-to-End Implementation | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) | [![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#) <br> [![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](#) |


## Progress

### Basic Components  
- [x] **Tokenization**
- [x] **Transformer Architecture**  
- [x] **Functions and Tools**
- [x] **Transformer Training**
- [ ] **Improvements and Experiments**

---

## Resources & References 📚  
| **Paper**                                                                 | **Component/Method**               | **Link**                                                                 |
|:-------------------------------------------------------------------------|:-----------------------------------|:-------------------------------------------------------------------------|
| Attention is All You Need                                                 | Transformer Architecture           | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)                     |
| Neural Machine Translation of Rare Words with Subword Units               | Byte-Pair Encoding (BPE)           | [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)                     |
| SentencePiece: A Simple Language-Independent Subword Tokenizer            | SentencePiece Tokenizer            | [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)                     |
| Layer Normalization                                                       | Layer Normalization                | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)                     |
| Adam: A Method for Stochastic Optimization                                | Adam Optimizer                     | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)                       |
| SGDR: Stochastic Gradient Descent with Warm Restarts                      | Cosine Learning Rate Scheduler     | [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)                     |
| Decoupled Weight Decay Regularization                                     | AdamW Optimizer                    | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)                     |
| Understanding the Difficulty of Training Deep Feedforward Neural Networks | Xavier/Glorot Initialization       | [PMLR 9:249-256](http://proceedings.mlr.press/v9/glorot10a)              |
| Rethinking the Inception Architecture for Computer Vision                 | Label Smoothing                    | [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)                     |
| Practical Bayesian Optimization of Machine Learning Algorithms            | Hyperparameter Tuning              | [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)                       |
| Beam Search Strategies for Neural Machine Translation                     | Beam Search Decoding               | [arXiv:1702.01806](https://arxiv.org/abs/1702.01806)                     |
| Efficient Transformers: A Survey                                          | Memory Optimization Techniques      | [arXiv:2009.06732](https://arxiv.org/abs/2009.06732)                     |

**Note**: This is a living project—code and structure may evolve.