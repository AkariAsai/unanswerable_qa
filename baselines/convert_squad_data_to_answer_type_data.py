import jsonlines
import os
from tqdm import tqdm
import argparse
import glob
import json
import gzip
import random
from collections import Counter


def create_squad_q_only_data(squad_data_fp, split="train"):
    squad_data = json.load(open(squad_data_fp))["data"]
    q_only_data = []
    for article in squad_data:
        paragraphs = article["paragraphs"]
        for paragraph in paragraphs:
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                example_id = qa["id"]
                label = "Answerable" if qa["is_impossible"] is False else "Unanswerable"
                if label == "Answerable" and split == "train" and random.random() > 0.5:
                    continue
                q_only_data.append(
                    {"question": question, "example_id": example_id, "label": label})
    print(
        "un-answerable examples # {}".format(Counter([item["label"] for item in q_only_data])))

    return q_only_data

def create_q_context_squad_data(squad_data_fp, split="train"):
    squad_data = json.load(open(squad_data_fp))["data"]
    q_context_data = []
    for article in squad_data:
        paragraphs = article["paragraphs"]
        for paragraph in paragraphs:
            qas = paragraph["qas"]
            context = paragraph["context"]
            
            for qa in qas:
                question = qa["question"]
                example_id = qa["id"]
                
                label = "Answerable" if qa["is_impossible"] is False else "Unanswerable"
                if label == "Answerable" and split == "train" and random.random() > 0.5:
                    continue
                q_context_data.append(
                    {"question": question, "paragraph": context, "example_id": example_id, "label": label})
    print(
        "un-answerable examples # {}".format(Counter([item["label"] for item in q_context_data])))

    return q_context_data
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_train_path",
                        default=None, type=str, required=True)
    parser.add_argument("--squad_dev_path",
                        default=None, type=str, required=True)
    parser.add_argument("--output_data_dir",
                        default=None, type=str, required=True)
    parser.add_argument('--question_only',
                        action='store_true')
    parser.add_argument('--qc_input',
                        action='store_true')
    args = parser.parse_args()

    # save data
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    if args.question_only is True:
        train_data = create_squad_q_only_data(args.squad_train_path, "train")
        dev_data = create_squad_q_only_data(args.squad_dev_path, "dev")
        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only=True_train.json"), "w") as outfile:
            json.dump(train_data, outfile)
        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only=True_dev.json"), "w") as outfile:
            json.dump(dev_data, outfile)

    elif args.qc_input is True:
        train_data = create_q_context_squad_data(args.squad_train_path, "train")
        dev_data = create_q_context_squad_data(args.squad_dev_path, "dev")
        with open(os.path.join(args.output_data_dir, "nq_answerable_q_c_train.json"), "w") as outfile:
            json.dump(train_data, outfile)
        with open(os.path.join(args.output_data_dir, "nq_answerable_q_c_dev.json"), "w") as outfile:
            json.dump(dev_data, outfile)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
