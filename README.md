# DeepCell: Self-Supervised Multiview Fusion for Circuit Representation Learning

Official code repository for the paper: 
[**DeepCell: Self-Supervised Multiview Fusion for Circuit Representation Learning**](https://arxiv.org/abs/2502.06816)

To appear in 2025 International Conference on Computer-Aided Design (ICCAD)

Authors: Zhengyuan Shi, Chengyu Ma, Ziyang Zheng, Lingfeng Zhou, Hongyang Pan, Wentao Jiang, Fan Yang, Xiaoyan Yang, Zhufei Chu, Qiang Xu

Contact: Zhengyuan Shi (zyshi21@cse.cuhk.edu.hk)

## Abstract
We introduce DeepCell, a novel circuit representation learning framework that effectively integrates multiview information from both And-Inverter Graphs (AIGs) and Post-Mapping (PM) netlists. At its core, DeepCell employs a self-supervised Mask Circuit Modeling (MCM) strategy, inspired by masked language modeling, to fuse complementary circuit representations from different design stages into unified and rich embeddings. To our knowledge, DeepCell is the first framework explicitly designed for PM netlist representation learning, setting new benchmarks in both predictive accuracy and reconstruction quality. We demonstrate the practical efficacy of DeepCell by applying it to critical EDA tasks such as functional Engineering Change Orders (ECO) and technology mapping. Extensive experimental results show that DeepCell significantly surpasses state-of-the-art open-source EDA tools in efficiency and performance.

## Install 
```sh
pip install -r requirements.txt
```

## Train
Train the AIG encoder and PM encoder separately 
```sh
bash ./run/aig_dg2.sh
bash ./run/pm_dg2.sh
```
You can choose other aggregator, see folder `./run`

Then, refine the embeddings of PM netlists using the AIG view. 
```sh
bash ./run/top.sh
```

All the pre-trained checkpoints can be found in `./ckpt`

## Citation
If DeepCell could help your project, please cite our work: 
```sh
@INPROCEEDINGS{10323798,
  author={Shi, Zhengyuan and Ma, Chengyu and Zheng, Ziyang and Zhou, Lingfeng and Pan, Hongyang and Jiang, Wentao and Yang, Fan and Yang, Xiaoyan and Chu, Zhufei and Xu, Qiang},
  booktitle={2025 IEEE/ACM International Conference on Computer Aided Design (ICCAD)}, 
  title={DeepCell: Self-Supervised Multiview Fusion for Circuit Representation Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-9}}
```