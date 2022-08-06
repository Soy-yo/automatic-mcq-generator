import abc

import numpy as np
from datasets import Dataset

from .dataset_preprocessor import BaseDatasetPreprocessor


class SQuADDataset(BaseDatasetPreprocessor, abc.ABC):
    _dataset_name = 'squad'
    _remove_columns = ['id', 'title', 'context', 'question', 'answers']


class SQuADAnswerGenerationDataset(SQuADDataset):

    def _preprocess(self, dataset, batch_size, num_proc):
        def concat(x):
            return np.unique(np.concatenate(list(x)))

        df = dataset.to_pandas()[['context', 'answers']]
        df['answers'] = df['answers'].map(lambda answer: answer['text'])
        data = Dataset.from_pandas(df.groupby('context').aggregate(concat))

        return data.map(self._preprocess_data).map(
            self._tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=['context', 'answers']
        )

    def _preprocess_data(self, example):
        return {
            'input_text': f"generate answers: {example['context']}",
            'target_text': self._tokenizer.eos_token.join(example['answers'])
        }


class SQuADQuestionGenerationDataset(SQuADDataset):

    def _preprocess(self, dataset, batch_size, num_proc):
        return dataset.map(self._preprocess_data).map(
            self._tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=self._remove_columns
        )

    @staticmethod
    def _preprocess_data(example):
        # Get first answer for example
        return {
            'input_text': f"generate question: {example['answers']['text'][0]} "
                          f"context: {example['context']}",
            'target_text': example['question']
        }


class SQuADAnsweringDataset(SQuADDataset):

    def _preprocess(self, dataset, batch_size, num_proc):
        return dataset.map(self._preprocess_data).map(
            self._tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=self._remove_columns
        )

    @staticmethod
    def _preprocess_data(example):
        # Get first answer for example
        return {
            'input_text': f"question: {example['question']} "
                          f"context: {example['context']}",
            'target_text': example['answers']['text'][0]
        }
