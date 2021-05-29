# Understanding Answerability in Information-Seeking Question Answering (EMNLP 2020 submission)

This repository contains codes for *Understanding Answerability in Information-Seeking Question Answering*. 

## Set up & Requirements
To run our code, please run the command below to install python packages.

```
pip -r install requirements.txt
```

## Download data
Please download data from the NQ website, TyDi QA repository and SQuAD 2.0 website. 

- [NQ](https://ai.google.com/research/NaturalQuestions/)
- [TyDi QA](https://github.com/google-research-datasets/tydiqa)
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) 

## Preprocess data
First you need to convert downloaded data into certain format to train binary or three-way answer type classifier. 

The commands to convert each dataset into the formats are shown below. Please changes `--type_num` to 2.0 for binary classification tasks. The default is 3.0 for NQ and TyDI QA.

### NQ 

```
python convert_nq_data_to_answer_type_data.py \
--nq_train_data_dir /PATH/TO/YOUR/NQ/TRAIN/DATA \
--nq_dev_data_dir /PATH/TO/YOUR/NQ/DEV/DATA \
--output_data_dir nq_q_only_two_way \
--question_only \
--type_num 2
```

### TyDi QA 
```
python convert_nq_data_to_answer_type_data.py \
--nq_train_data_dir /PATH/TO/YOUR/TYDI/TRAIN/DATA \
--nq_dev_data_dir/PATH/TO/YOUR/TYDI/DEV/DATA \
--output_data_dir tydi_q_only_two_way \
--tydi \
```

### SQuAD 2.0

```
python convert_nq_data_to_answer_type_data.py \
--squad_train_path /PATH/TO/YOUR/SQUAD/TRAIN/DATA \
--squad_dev_path /PATH/TO/YOUR/SQUAD/DEV/DATA\
--output_dir sqaud_q_only_two_way \
--question_only 
```


## Train and Evaluate models
Run the command below to train and evaluate models. After running evaluations, the detailed results with `accuracy`, `preds` (prediction results) and `output`(ground-truth labels) will be stored in `/PATH/TO/YOUR/OUTPUT/DIR/eval_outputs_results.json`

### NQ 

```
python run_q_only_answer_type_classification.py \
--data_dir nq_q_only_two_way \
--model_type bert \
--model_name_or_path bert-base-uncased \
--do_lower_case \ 
--task_name answer_type_q_only_two \
--output_dir output_nq_q_only_two_way \
--do_train --do_eval \
--per_gpu_train_batch_size 8
```

### TyDi QA 
```
python run_q_only_answer_type_classification.py \
--data_dir tydi_q_only_two_way \
--model_type bert \
--do_train --do_eval \
--model_name_or_path bert-base-multilingual-uncased \
--do_lower_case \ 
--task_name answer_type \
--output_dir output_tydi_q_only_three \
--max_seq_length 126 \
--do_lower_case --per_gpu_train_batch_size 8 
```

### SQuAD 2.0

```
python run_q_only_answer_type_classification.py \
--data_dir sqaud_q_only_two_way \
--model_type bert \
--model_name_or_path bert-base-uncased \
--do_lower_case \ 
--task_name answer_type_q_only_two \
--output_dir output_squad_q_only_two_way \
--do_train --do_eval \
--per_gpu_train_batch_size 8
```


