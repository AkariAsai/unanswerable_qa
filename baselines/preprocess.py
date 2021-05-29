""" Modal classification baseline for the Multi-modal Multi-hop QA dataset."""

import logging
import os
import json

from utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" %
                        (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            if example.label not in label_map:
                print("this label does not exist")
                print(example)
                continue
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features


class AnswerTypeClassifier(DataProcessor):
    """Processor for modality classification dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "answer_type_classification_question_only=True_train.json"))), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "answer_type_classification_question_only=True_dev.json"))), "dev")

    def get_labels(self):
        """See base class."""
        return ["Short", "Long", "None"]

    def _create_examples(self, orig_examples, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in orig_examples:
            example_id = str(example["example_id"])
            
            text_a = example["question"]
            label = example["label"]
            examples.append(InputExample(
                guid=example_id, text_a=text_a, text_b=None, label=label))
        return examples


class AnswerTypeClassifierTwo(DataProcessor):
    """Processor for modality classification dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "answer_type_classification_question_only=True_train.json"))), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "answer_type_classification_question_only=True_dev.json"))), "dev")

    def get_labels(self):
        """See base class."""
        return ["Unanswerable", "Answerable"]

    def _create_examples(self, orig_examples, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in orig_examples:
            example_id = str(example["example_id"])
            
            text_a = example["question"]
            label = example["label"]
            examples.append(InputExample(
                guid=example_id, text_a=text_a, text_b=None, label=label))
        return examples
    

class AnswerTypeClassifierContext(DataProcessor):
    """Processor for modality classification dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "nq_answerable_q_c_train.json"))), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "nq_answerable_q_c_dev.json"))), "dev")

    def get_labels(self):
        """See base class."""
        return ["Short", "Long", "None"]

    def _create_examples(self, orig_examples, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in orig_examples:
            example_id = str(example["example_id"])
            
            text_a = example["question"]
            text_b = example["paragraph"]
            label = example["label"]
            
            examples.append(InputExample(
                guid=example_id, text_a=text_a, text_b=text_b, label=label))
        return examples


class AnswerTypeClassifierContextTwo(DataProcessor):
    """Processor for modality classification dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "nq_answerable_q_c_train.json"))), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(json.load(open(os.path.join(data_dir, "nq_answerable_q_c_dev.json"))), "dev")

    def get_labels(self):
        """See base class."""
        return ["Unanswerable", "Answerable"]

    def _create_examples(self, orig_examples, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in orig_examples:
            example_id = str(example["example_id"])
            
            text_a = example["question"]
            text_b = example["paragraph"]
            label = example["label"]
            
            examples.append(InputExample(
                guid=example_id, text_a=text_a, text_b=text_b, label=label))
        return examples
    
def answer_type_compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return (preds == labels).mean()


answer_type_tasks_num_labels = {
    "answer_type": 3,
    "answer_type_c": 3,
    "answer_type_q_only_two":2,
    "answer_type_c_two": 2,
}

answer_type_processors = {
    "answer_type": AnswerTypeClassifier,
    "answer_type_c": AnswerTypeClassifierContext,
    "answer_type_q_only_two": AnswerTypeClassifierTwo,
    "answer_type_c_two": AnswerTypeClassifierContextTwo,
}

answer_type_output_modes = {
    "answer_type": "classification",
    "answer_type_c": "classification",
    "answer_type_q_only_two": "classification",
    "answer_type_c_two": "classification",
}
