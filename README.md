# Code for "Challenges in Information Seeking QA:Unanswerable Questions and Paragraph Retrieval" (ACL 2021, Long)

This is the repository for baseline models and annotated data for this paper:
Akari Asai and Eunsol Choi. Challenges in Information Seeking QA:Unanswerable Questions and Paragraph Retrieval. In: Proceedings of ACL. 2021

In the paper, we carefully analyze unanswerable questions in information-seeking QA dataset (i.e., Natural Questions and TyDi QA) and attempt to identify the remaining headrooms. We conduct both a range of controlled experiments and insensitive human annotations on around 800 examples across across 6 languages.

## Annotated data
In [`human_annotated_data`](human_annotated_data), we provide human annotated data from TyDi QA and Natural Questions. 

| Dataset      | language | # of annotated questions | file name | 
| ----------- | ----------- |----------- |----------- |
| Natural Questions    | English | 450 | [NQ.tsv](human_annotated_data/NQ.tsv) |
| TyDi QA    | Bengali | 50 | [TyDi-Bn.tsv](human_annotated_data/TyDi-Bn.tsv) |
| TyDi QA      | Japanese | 100 | [TyDi-Ja.tsv](human_annotated_data/TyDi-Ja.tsv) |
| TyDi QA     | Korean | 100 | [TyDi-Bn.tsv](human_annotated_data/TyDi-Ko.tsv) |
| TyDi QA     | Russian | 50 | [TyDi-Ru.tsv](human_annotated_data/TyDi-Ru.tsv) |
| TyDi QA     | Telugu | 50 | [TyDi-Te.tsv](human_annotated_data/TyDi-Te.tsv) |


## Baselines
In this work, we conduct several baseline experiments to identify the remaining headrooms in information-seeking QA. This repository include baselines for question only baseline. See the training and evaluation details in [README.md](baselines/README.md). We thank the authors of [ Riki Net](https://arxiv.org/abs/2004.14560), [Retro-reader](https://arxiv.org/abs/2001.09694), and [ETC](https://arxiv.org/abs/2004.08483) for providing their models' predictions that are used to analyze those state-of-the-art models behaviors.


Citation and Contact
If you find this codebase is useful or use in your work, please cite our paper.

@inproceedings{
asai2020learning,
title={Challenges in Information Seeking QA:Unanswerable Questions and Paragraph Retrieval},
author={Akari Asai and Eunsol Choi},
booktitle={ACL-IJCNLP},
year={2021}
}
Please contact Akari Asai (@AkariAsai, akari[at]cs.washington.edu) for questions and suggestions.
