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
    
def create_graph_retriever_train_dev_data(original_nq_data_dir, split="dev"):
    graph_retriever_data = []
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        with jsonlines.open(fn) as reader:
            for obj in tqdm(reader):
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                long_answer_candidates = obj["long_answer_candidates"]
                documents_tokens = [token["token"] for token in obj["document_tokens"]]
                
                long_answers = []
                null_count = 0
                for annotation in annotations:
                    if annotation["long_answer"]["candidate_index"] > -1:
                        long_answers.append(
                            annotation["long_answer"])
                    else:
                        null_count += 1
                if (split == "train" and len(long_answers) > 0) or (split == "train" and null_count <= 3):
                    continue
                else:
                    # create context dict
                    context = {}
                    for long_answer_idx, long_answer_candidate in enumerate(long_answer_candidates):
                        if documents_tokens[long_answer_candidate["start_token"]].lower() == "<p>":
                            context[long_answer_idx] = " ".join(
                                documents_tokens[long_answer_candidate["start_token"]:long_answer_candidate["end_token"]])
                    new_data = {"question": question, "q_id": example_id, "context": context,
                                'all_linked_para_title_dic': {}, 'all_linked_paras_dic': {}, 'short_gold': [], 'redundant_gold': [], 'all_redundant_gold': []}
                    graph_retriever_data.append(new_data)
                
        print("{} examples added".format(len(graph_retriever_data)))
        print(graph_retriever_data[-1])

    return graph_retriever_data


def create_graph_retriever_train_dev_data_full(original_nq_data_dir, split="dev"):
    graph_retriever_data = []
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        with jsonlines.open(fn) as reader:
            for obj in tqdm(reader):
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                long_answer_candidates = obj["long_answer_candidates"]
                documents_tokens = [token["token"]
                                    for token in obj["document_tokens"]]

                long_answers = []
                null_count = 0
                for annotation in annotations:
                    if annotation["long_answer"]["candidate_index"] > -1:
                        long_answers.append(
                            annotation["long_answer"])
                    else:
                        null_count += 1
                # create context dict
                context = {}
                for long_answer_idx, long_answer_candidate in enumerate(long_answer_candidates):
                    context[long_answer_idx] = " ".join(
                        documents_tokens[long_answer_candidate["start_token"]:long_answer_candidate["end_token"]])
                print(len(context))
                new_data = {"question": question, "q_id": example_id, "context": context,
                            'all_linked_para_title_dic': {}, 'all_linked_paras_dic': {}, 'short_gold': [], 'redundant_gold': [], 'all_redundant_gold': []}
                graph_retriever_data.append(new_data)

        print("{} examples added".format(len(graph_retriever_data)))
        print(graph_retriever_data[-1])

    return graph_retriever_data


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



def create_long_short_classification_data_w_gold_para(original_nq_data_dir, split):
    input_data = []
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        example_num, long_answer_num, short_answer_num, none_answer_num = 0, 0, 0, 0
        with jsonlines.open(fn) as reader:
            for obj in reader:
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                document_url = obj["document_url"]
                long_answer_candidates = obj["long_answer_candidates"]
                documents_tokens = [token["token"] for token in obj["document_tokens"]]
                
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
                        
                if (split == "train" and len(long_answers) == 0) or (split == "dev" and null_count > 3):
                    none_answer_num += 1
                    continue
                    
                elif (split == "train" and len(long_answers) > 0) or (split == "dev" and null_count <= 3) and len(short_answers) == 0:
                    long_answer = long_answer_candidates[long_answers[0]["candidate_index"]]    
                    gold_paragraph = " ".join(
                        documents_tokens[long_answer["start_token"]:long_answer["end_token"]])
                    
                    input_data.append(
                        {"example_id": example_id, "question": question, "paragraph": gold_paragraph, "label": "Long", "document_url": document_url})
                    long_answer_num += 1
                    
                elif(split == "train" and len(long_answers) > 0) or (split == "dev" and null_count <= 3) and len(short_answers) > 0:
                    long_answer = long_answer_candidates[long_answers[0]
                                                         ["candidate_index"]]
                    gold_paragraph = " ".join(
                        documents_tokens[long_answer["start_token"]:long_answer["end_token"]])

                    input_data.append(
                        {"example_id": example_id, "question": question, "paragraph": gold_paragraph, "label": "Short", "document_url": document_url})

                    short_answer_num += 1
                else:
                    print(short_answers)
                    print(long_answers)
                    raise NotImplementedError()
                example_num += 1

        print("{} examples added".format(example_num))
        print("{0} long examples, {1} short examples and {2} none examples".format(
            long_answer_num, short_answer_num, none_answer_num))
        print(input_data[-1])

    return input_data


def create_long_short_classification_model_final_output(original_nq_data_dir, pred_file):
    input_data = []
    model_pred = json.load(open(pred_file))["predictions"]
    model_pred_dic = {pred["example_id"]: pred for pred in model_pred}
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        with jsonlines.open(fn) as reader:
            for obj in reader:
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                document_url = obj["document_url"]
                documents_tokens = [token["token"]
                                    for token in obj["document_tokens"]]

                long_answers = []
                short_answers = []
                non_null_count = 0
                for annotation in annotations:
                    if annotation["long_answer"]["candidate_index"] > -1:
                        long_answers.append(
                            annotation["long_answer"])
                        non_null_count += 1
                    if annotation["short_answers"] != []:
                        short_answers.append(annotation["short_answers"])
                    if annotation["yes_no_answer"] != "NONE":
                        short_answers.append(annotation["yes_no_answer"])
                        
                # decide label
                if non_null_count < 2:
                    label = "None"
                elif non_null_count >= 2 and len(short_answers) == 0:
                    label = "Long"
                elif non_null_count >= 2 and len(short_answers) > 0:
                    label = "Short"
                else:
                    raise NotImplementedError()
                
                # store the document tokens from RikiNet Predictions 
                start_token = model_pred_dic[example_id]["long_answer"]["start_token"]
                end_token = model_pred_dic[example_id]["long_answer"]["end_token"]
                pred_paragraph = " ".join(documents_tokens[start_token:end_token])
                input_data.append(
                    {"example_id": example_id, "question": question, "paragraph": pred_paragraph, "label": label, "document_url": document_url})

    assert len(input_data) == 7830
    # show dataset statistics
    long_answer_num, short_answer_num, none_answer_num = 0, 0, 0
    for example in input_data:
        if example["label"] == "Short":
            short_answer_num += 1
        elif example["label"] == "Long":
            long_answer_num += 1
        elif example["label"] == "None":
            none_answer_num += 1
        else:
            raise NotImplementedError()
    print(
        "THe number of data: short --> {0}, long --> {1}, null --> {2}".format(short_answer_num, long_answer_num, none_answer_num))
    print(input_data[-1])

    return input_data

# Use randomly sampled train data for question-context type classification
def create_long_short_classification_data_w_negative_examples_train(original_nq_data_dir):
    input_data = []
    for fn in glob.glob(os.path.join(original_nq_data_dir, "*.jsonl")):
        print("loading {}...".format(fn))
        example_num, long_answer_num, short_answer_num, none_answer_num = 0, 0, 0, 0
        with jsonlines.open(fn) as reader:
            for obj in reader:
                question = obj["question_text"]
                example_id = obj["example_id"]
                annotations = obj["annotations"]
                document_url = obj["document_url"]
                long_answer_candidates = obj["long_answer_candidates"]
                documents_tokens = [token["token"]
                                    for token in obj["document_tokens"]]

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

                if len(long_answers) == 0:
                    for long_answer_candidate in long_answer_candidates:
                        # Following Pan et al. (2019).
                        if random.random() < 0.01:
                            gold_paragraph = " ".join(
                                documents_tokens[long_answer_candidate["start_token"]:long_answer_candidate["end_token"]])
                            input_data.append(
                                {"example_id": example_id, "question": question, "paragraph": gold_paragraph, "label": "None", "document_url": document_url})
                            none_answer_num += 1
                else:
                    for cand_i, long_answer_candidate in enumerate(long_answer_candidates):
                        if cand_i in [long_answer["candidate_index"] for long_answer in long_answers]:
                            gold_paragraph = " ".join(
                                documents_tokens[long_answer_candidate["start_token"]:long_answer_candidate["end_token"]])
                            if len(short_answers) > 0:
                                input_data.append({"example_id": example_id, "question": question, "paragraph": gold_paragraph, "label": "Short", "document_url": document_url})
                                short_answer_num += 1
                            else:
                                input_data.append(
                                    {"example_id": example_id, "question": question, "paragraph": gold_paragraph, "label": "Long", "document_url": document_url})
                                long_answer_num += 1
                        else:
                            if random.random() < 0.005:
                                distractor_paragraph= " ".join(documents_tokens[long_answer_candidate["start_token"]:long_answer_candidate["end_token"]])
                                input_data.append({"example_id": example_id, "question": question,
                                                           "paragraph": distractor_paragraph, "label": "None", "document_url": document_url})
                                none_answer_num += 1

        print("{} examples added".format(len(input_data)))
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
    parser.add_argument("--riki_pred_file",
                        default=None, type=str)
    parser.add_argument('--question_only',
                        action='store_true')
    parser.add_argument('--graph_rt',
                        action='store_true')
    parser.add_argument('--qc_input',
                        action='store_true')
    parser.add_argument('--rikinet',
                        action='store_true')
    parser.add_argument('--negative_sample_train',
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
        
    elif args.rikinet is True:
        dev_data = create_long_short_classification_model_final_output(
            args.nq_dev_data_dir, args.riki_pred_file)
        with open(os.path.join(args.output_data_dir, "answer_type_classification_rikinet_dev.json"), "w") as outfile:
            json.dump(dev_data, outfile)
            
    elif args.negative_sample_train is True:
        train_data = create_long_short_classification_data_w_negative_examples_train(
            args.nq_train_data_dir)
        with open(os.path.join(args.output_data_dir, "answer_type_classification_qc_negative_sampling"), "w") as outfile:
            json.dump(train_data, outfile)
    # Create classification data
    elif args.question_only is True:
        train_data = create_long_short_classification_data(args.nq_train_data_dir, "train", args.type_num)
        dev_data = create_long_short_classification_data(
            args.nq_dev_data_dir, "dev", args.type_num)
        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only={}_train.json".format(args.question_only)), "w") as outfile:
            json.dump(train_data, outfile)

        with open(os.path.join(args.output_data_dir, "answer_type_classification_question_only={}_dev.json".format(args.question_only)), "w") as outfile:
            json.dump(dev_data, outfile)

    elif args.graph_rt is True:
        train_data = create_graph_retriever_train_dev_data(
            args.nq_train_data_dir, "train")
        dev_data = create_graph_retriever_train_dev_data_full(
            args.nq_dev_data_dir, "dev")
        with open(os.path.join(args.output_data_dir, "nq_unanswerable_graph_train.json"), "w") as outfile:
            json.dump(train_data, outfile)

        with open(os.path.join(args.output_data_dir, "nq_unanswerable_graph_dev.json"), "w") as outfile:
            json.dump(dev_data, outfile)

    elif args.qc_input is True:
        train_data = create_long_short_classification_data_w_gold_para(
            args.nq_train_data_dir, "train")
        dev_data = create_long_short_classification_data_w_gold_para(
            args.nq_dev_data_dir, "dev")
        with open(os.path.join(args.output_data_dir, "nq_answerable_q_c_train.json"), "w") as outfile:
            json.dump(train_data, outfile)

        with open(os.path.join(args.output_data_dir, "nq_answerable_q_c_dev.json"), "w") as outfile:
            json.dump(dev_data, outfile)
        
    else:
        raise NotImplementedError()
    



if __name__ == "__main__":
    main()


