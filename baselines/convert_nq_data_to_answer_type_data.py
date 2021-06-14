import jsonlines
import os
from tqdm import tqdm 
import argparse
import glob 
import json
import gzip
import random


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj) 
    return lines

def create_tydiqa_q_only_data(original_tydi_data, split="train", type_num=3):
    # NULL --> if less than two out of three annotators select any answer, the questions should be NULL.
    unanswerable_questions = []
    examples = read_jsonlines(original_tydi_data)
    long_answer_num, short_answer_num, none_answer_num, answerable_num = 0, 0, 0, 0
    for example in tqdm(examples):
        lang = example["language"]
        annot = example["annotations"]
        question_text = example["question_text"]
        example_id = example["example_id"]
        document_url = example["document_url"]

        if split == "train":
            if annot[0]["passage_answer"]["candidate_index"] == -1:
                none_answer_num += 1
                if type_num == 2:
                    label = "Unanswerable"
                else:
                    label = "None"
                    
            else:
                answerable_num += 1
                if type_num == 2:
                    label = "Answerable"
                else:
                    if  annot[0]["minimal_answer"]["plaintext_end_byte"] == -1:
                        label = "Long"
                        long_answer_num += 1
                    else:
                        label = "Short"
                        short_answer_num += 1
                        
        elif split == "dev":
            if len([annot_i for annot_i in annot if annot_i["passage_answer"]["candidate_index"] == -1]) >= 2:
                none_answer_num += 1
                if type_num == 2:
                    label = "Unanswerable"
                else:
                    label = "None"

            else:
                answerable_num += 1
                if type_num == 2:
                    label = "Answerable"
                else:
                    if len([annot_i for annot_i in annot if annot_i["minimal_answer"]["plaintext_end_byte"] > -1]) > 0:
                        label = "Short"
                        short_answer_num += 1
                    else:
                        label = "Long"
                        long_answer_num += 1
                
        unanswerable_questions.append(
            {"example_id": example_id, "question": question_text, "label": label, "document_url": document_url, "lang": lang})
    print("split:{}".format(split))
    print("{} examples added".format(none_answer_num))
    print("{0} long examples, {1} short examples (total: {2}) and {3} none examples".format(
        long_answer_num, short_answer_num, answerable_num, none_answer_num))
    if type_num == 2:
        assert answerable_num + none_answer_num == len(examples)
    elif type_num == 3:    
        assert short_answer_num + long_answer_num == answerable_num
        assert answerable_num + none_answer_num == len(examples)
    print(unanswerable_questions[-1])
        
    return unanswerable_questions

def create_long_short_classification_data(original_nq_data_dir, split="train", type_num=3):
    input_data = []
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        example_num, long_answer_num, short_answer_num, none_answer_num, answerable_num = 0, 0, 0, 0, 0
        with jsonlines.open(fn) as reader:
            for obj in reader:
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                document_url = obj["document_url"]
                long_answers = []
                short_answers = []
                null_count = 0
                for annotation in annotations:
                    if annotation["long_answer"]["candidate_index"] > -1:
                        long_answers.append(
                            annotation["long_answer"])
                    else:
                        null_count += 1
                    if annotation["short_answers"] != []:
                        short_answers.append(annotation["short_answers"])
                    if annotation["yes_no_answer"] != "NONE":
                        short_answers.append(annotation["yes_no_answer"])
                if split == "train":
                    if len(long_answers) == 0:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "None" if type_num == 3 else "Unanswerable", "document_url": document_url})
                        none_answer_num += 1
                    elif len(long_answers) > 0 and len(short_answers) == 0:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "Long" if type_num == 3 else "Answerable", "document_url": document_url})
                        long_answer_num += 1
                        answerable_num += 1
                    elif len(long_answers) > 0 and len(short_answers) > 0:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "Short" if type_num == 3 else "Answerable", "document_url": document_url})
                        short_answer_num += 1
                        answerable_num += 1
                    else:
                        print(short_answers)
                        print(long_answers)
                        raise NotImplementedError()
                    example_num += 1
                else:
                    if len(long_answers) == 0 or null_count > 3:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "None" if type_num == 3 else "Unanswerable", "document_url": document_url})
                        none_answer_num += 1
                    elif len(long_answers) > 1 and null_count <= 3 and len(short_answers) == 0:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "Long" if type_num == 3 else "Answerable", "document_url": document_url})
                        long_answer_num += 1
                        answerable_num += 1
                    elif len(long_answers) > 1 and null_count <= 3 and len(short_answers) > 0:
                        input_data.append(
                            {"example_id": example_id, "question": question, "label": "Short" if type_num == 3 else "Answerable", "document_url": document_url})
                        short_answer_num += 1
                        answerable_num += 1
                    else:
                        print(short_answers)
                        print(long_answers)
                        raise NotImplementedError()
                    example_num += 1
        # Per file log
        print("{} examples added".format(example_num))
        assert short_answer_num + long_answer_num == answerable_num
        assert answerable_num + none_answer_num == example_num
        print("{0} long examples, {1} short examples and {2} none examples".format(
            long_answer_num, short_answer_num, none_answer_num))
        print(input_data[-1])

    return input_data

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq_train_data_dir",
                        default=None, type=str, required=True)
    parser.add_argument("--nq_dev_data_dir",
                        default=None, type=str, required=True)
    parser.add_argument("--output_data_dir",
                        default=None, type=str, required=True)
    parser.add_argument("--type_num",
                        default=3, type=int)
    parser.add_argument('--question_only',
                        action='store_true')
    parser.add_argument('--tydi',
                        action='store_true')
    args = parser.parse_args()
    
    # save data
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    if args.tydi is True:
        train_data = create_tydiqa_q_only_data(
            args.nq_train_data_dir, "train", args.type_num)
        dev_data = create_tydiqa_q_only_data(
            args.nq_dev_data_dir, "dev", args.type_num)
        with open(os.path.join(args.output_data_dir, "tydi_train_q_only.json"), "w") as outfile:
            json.dump(train_data, outfile)
        with open(os.path.join(args.output_data_dir, "tydi_dev_q_only.json"), "w") as outfile:
            json.dump(dev_data, outfile)

    elif args.question_only is True:
        train_data = create_long_short_classification_data(args.nq_train_data_dir, "train", args.type_num)
        dev_data = create_long_short_classification_data(
            args.nq_dev_data_dir, "dev", args.type_num)
        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only={}_train.json".format(args.question_only)), "w") as outfile:
            json.dump(train_data, outfile)

        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only={}_dev.json".format(args.question_only)), "w") as outfile:
            json.dump(dev_data, outfile)

    else:
        raise NotImplementedError()



if __name__ == "__main__":
    main()


