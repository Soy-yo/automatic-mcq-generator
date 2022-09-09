import re
import abc
import string
from collections import Counter
from argparse import ArgumentParser

import torch
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric

from mcq.data.multitask_mcq import MultitaskMCQDataset


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


BAR_TEMPLATE = '[{}] {:.2f} %'


def print_bar(i, n, bar_size=20):
    progress = int(np.ceil(bar_size * i / n))
    bar_content = '#' * progress + '-' * (bar_size - progress)

    end = '' if i < n else '\n'

    print(f'\r{BAR_TEMPLATE.format(bar_content, i / n * 100)}', end=end)


class CustomMetric(abc.ABC):

    def __init__(self, name):
        self._buffer = []
        self.name = name

    def add(self, *, predictions=None, references=None):
        assert predictions is not None and references is not None
        self._buffer.append((predictions, references))

    def add_batch(self, *, predictions=None, references=None):
        assert predictions is not None and references is not None
        for pred, ref in zip(predictions, references):
            self._buffer.append((pred, ref))

    def compute(self, *, predictions=None, references=None):
        if predictions is None or references is None:
            return self._compute_buffer()

    def _compute_buffer(self):
        total = 0
        count = 0
        for pred, ref in self._buffer:
            total += self._compute(predictions=pred, references=ref)[self.name]
            count += 1

        return {self.name: total / count}

    @abc.abstractmethod
    def _compute(self, predictions, references):
        pass


# Adapted from
# https://huggingface.co/spaces/evaluate-metric/squad/blob/main/compute_score.py
class ExactMatchMetric(CustomMetric):

    def __init__(self):
        super(ExactMatchMetric, self).__init__('exact_match')

    def _compute(self, predictions, references):
        exact_match = float(
            metric_max_over_ground_truths(
                self._exact_match_score,
                predictions,
                references
            )
        )

        return {self.name: exact_match}

    @staticmethod
    def _exact_match_score(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)


class F1Metric(CustomMetric):

    def __init__(self):
        super(F1Metric, self).__init__('f1')

    def _compute(self, predictions, references):
        f1 = metric_max_over_ground_truths(
            self._f1_score,
            predictions,
            references
        )
        return {'f1': f1}

    @staticmethod
    def _f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class TaskEvaluator:

    def __init__(self, model, tokenizer, data, multi=False):
        self._model = model
        self._tokenizer = tokenizer
        self._multi = multi
        self._data = data

    def evaluate(self, em=True, f1=True, bleu=True, **kwargs):
        assert any((em, f1, bleu))

        em_metric = ExactMatchMetric() if em else None
        f1_metric = F1Metric() if f1 else None
        bleu_metric = load_metric('bleu') if bleu else None

        n = len(self._data)

        print(
            f'Starting evaluation for {n} examples on the following metrics: ',
            end=''
        )
        print(', '.join([
            metric.name
            for metric in (em_metric, f1_metric, bleu_metric)
            if metric is not None
        ]))

        for i, x in enumerate(self._data):
            print_bar(i, n)

            predictions = self._generate(x, **kwargs)
            references = torch.tensor(x['labels'])

            if self._multi:
                predictions = self._preprocess(predictions)
                references = self._preprocess(references)
            else:
                predictions = [self._remove_invalid_tokens(predictions)]
                references = [self._remove_invalid_tokens(references)]

            def decode(seqs):
                return [
                    self._tokenizer.decode(s, skip_special_tokens=True)
                    for s in seqs
                ]

            decoded_predictions = None
            decoded_references = None

            if em_metric is not None:
                decoded_predictions = decode(predictions)
                decoded_references = decode(references)
                for pred in decoded_predictions:
                    em_metric.add(
                        predictions=pred,
                        references=decoded_references
                    )

            if f1_metric is not None:
                if decoded_predictions is None:
                    decoded_predictions = decode(predictions)
                    decoded_references = decode(references)
                for pred in decoded_predictions:
                    f1_metric.add(
                        predictions=pred,
                        references=decoded_references
                    )

            if bleu_metric is not None:
                for pred in predictions:
                    bleu_metric.add(predictions=pred, references=references)

        result = {
            metric.name: metric.compute()[metric.name]
            for metric in (em_metric, f1_metric, bleu_metric)
            if metric is not None
        }

        print_bar(n, n)

        return result

    def _generate(self, x, **kwargs):
        if self._multi:
            return self._multi_generate(x, **kwargs)
        else:
            return self._model.generate(
                input_ids=torch.tensor([x['input_ids']]),
                attention_mask=torch.tensor([x['attention_mask']]),
                **kwargs
            )[0]

    def _multi_generate(self, x, **kwargs):
        n = 3 if self._multi is True else self._multi

        predictions = []
        decoder_input_ids = [self._tokenizer.pad_token_id]
        for k in range(n):
            predictions = self._model.generate(
                input_ids=torch.tensor([x['input_ids']]),
                attention_mask=torch.tensor([x['attention_mask']]),
                decoder_input_ids=torch.tensor([decoder_input_ids]),
                **kwargs
            )[0]
            decoder_input_ids = list(predictions)

        return predictions

    def _remove_invalid_tokens(self, seq):
        return seq[(seq > 0) & (seq != self._tokenizer.pad_token_id)]

    def _split(self, seq):
        indices, = np.where(seq == self._tokenizer.eos_token_id)
        sequences = np.split(seq, indices + 1)
        return [s for s in sequences if len(s) > 0]

    def _preprocess(self, seq):
        return self._split(self._remove_invalid_tokens(seq))


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        'model_name',
        type=str,
        help='Location of the model to be evaluated.'
    )
    parser.add_argument(
        '-t', '--tokenizer',
        type=str,
        default='t5-small',
        help='Name of the T5 tokenizer to be used.'
    )
    parser.add_argument(
        '-a', '--ag',
        action='store_true',
        help='Evaluate AG.'
    )
    parser.add_argument(
        '-q', '--qg',
        action='store_true',
        help='Evaluate QG.'
    )
    parser.add_argument(
        '-d', '--dg',
        action='store_true',
        help='Evaluate DG.'
    )

    return parser.parse_args()


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    dataset = MultitaskMCQDataset(tokenizer)

    if args.ag:
        ag_data = dataset.get_squad_answer_generation_data(subsets=['test'])
        print('ANSWER GENERATION')
        evaluator = TaskEvaluator(model, tokenizer, ag_data['test'], multi=True)
        print(evaluator.evaluate(max_new_tokens=20))
        print()

    if args.qg:
        qg_data = dataset.get_squad_question_generation_data(subsets=['test'])
        print('QUESTION GENERATION')
        evaluator = TaskEvaluator(model, tokenizer, qg_data['test'])
        print(evaluator.evaluate(max_new_tokens=30))
        print()

    if args.dg:
        dg_data = dataset.get_race_distractor_generation_data(subsets=['test'])
        print('DISTRACTOR GENERATION')
        evaluator = TaskEvaluator(model, tokenizer, dg_data['test'], multi=True)
        print(evaluator.evaluate(max_new_tokens=20))
        print()


if __name__ == '__main__':
    main()
