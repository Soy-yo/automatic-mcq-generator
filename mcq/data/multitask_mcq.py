from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

from .preprocessors import (
    SQuADAnswerGenerationPreprocessor,
    SQuADQuestionGenerationPreprocessor,
    RACEDistractorGenerationPreprocessor
)


class MultitaskMCQDataset:
    """MCQG dataset generator from an SQuAD and RACE combination."""

    _default_subsets = ('train', 'validation')

    def __init__(
            self,
            tokenizer,
            ag_preprocessor=None,
            qg_preprocessor=None,
            dg_preprocessor=None
    ):
        """Constructs a MultitaskMCQGDataset.

        Parameters
        ----------
        tokenizer : transformers.T5Tokenizer
            Tokenizer used to encode the texts.

        ag_preprocessor : mcq.data.ExamplePreprocessor or callable
            Function that accept an example and returns a dict with input_text
            and target_text for answer generation.

        qg_preprocessor : mcq.data.ExamplePreprocessor or callable
            Function that accept an example and returns a dict with input_text
            and target_text for question generation.

        dg_preprocessor : mcq.data.ExamplePreprocessor or callable
            Function that accept an example and returns a dict with input_text
            and target_text for distractor generation.
        """

        if ag_preprocessor is None:
            ag_preprocessor = SQuADAnswerGenerationPreprocessor(tokenizer)
        if qg_preprocessor is None:
            qg_preprocessor = SQuADQuestionGenerationPreprocessor(tokenizer)
        if dg_preprocessor is None:
            dg_preprocessor = RACEDistractorGenerationPreprocessor(tokenizer)

        self._tokenizer = tokenizer
        self._squad = load_dataset('squad')
        self._race = load_dataset('race', 'all')
        self._ag_preprocessor = ag_preprocessor
        self._qg_preprocessor = qg_preprocessor
        self._dg_preprocessor = dg_preprocessor

    def get_multitask(
            self,
            batch_size=1024,
            max_input_length=None,
            max_target_length=None,
            subsets=None,
            num_proc=None
    ):
        """Returns the multitask dataset by combining SQuAD and RACE datasets.

        Parameters
        ----------
        batch_size : int
            Batch size used to generate encodings with the tokenizer. Defaults
            to 1024.

        max_input_length : int or dict[str, int]
            Max length of input encodings. If dict, its keys must be 'ag', 'qg'
            and 'dg' and their values the respective amx input lengths for each
            task.

        max_target_length : int or dict[str, int]
            Max length of target encodings. If dict, its keys must be 'ag', 'qg'
            and 'dg' and their values the respective amx target lengths for each
            task.

        subsets : list[str]
            Dataset subsets to return. Valid values are 'train', 'validation'
            and 'test'. Defaults to ('train', 'validation'). For SQuAD, half
            validation set will be used as test set, since it does not come with
            it, but the validation set is twice as big as RACE's.

        num_proc : int or None
            Number of processes to be used to transform data.

        Returns
        -------
        DatasetDict
            Generated dataset with requested subsets.

        """

        kwargs = {
            'batch_size': batch_size,
            'subsets': subsets,
            'num_proc': num_proc
        }
        datasets = []

        for name, fn in (
                ('ag', self.get_squad_answer_generation_data),
                ('qg', self.get_squad_question_generation_data),
                ('dg', self.get_race_distractor_generation_data)
        ):
            if max_input_length is not None:
                if isinstance(max_input_length, int):
                    kwargs['max_input_length'] = max_input_length
                elif max_input_length.get(name) is None:
                    kwargs.pop('max_input_length', None)
                else:
                    kwargs['max_input_length'] = max_input_length[name]

            if max_target_length is not None:
                if isinstance(max_target_length, int):
                    kwargs['max_target_length'] = max_target_length
                elif max_target_length.get(name) is None:
                    kwargs.pop('max_target_length', None)
                else:
                    kwargs['max_target_length'] = max_target_length[name]

            dataset = fn(**kwargs)

            datasets.append(dataset)

        dataset_dict = defaultdict(lambda: [])
        for dataset in datasets:
            for subset in dataset.keys():
                dataset_dict[subset].append(dataset[subset])

        return DatasetDict({
            subset: concatenate_datasets(ds)
            for subset, ds in dataset_dict.items()
        })

    def get_squad_answer_generation_data(
            self,
            batch_size=1024,
            max_input_length=512,
            max_target_length=256,
            subsets=None,
            num_proc=None
    ):
        """Returns answer generation dataset only from SQuAD.

        Parameters
        ----------
        batch_size : int
            Batch size used to generate encodings with the tokenizer. Defaults
            to 1024.

        max_input_length : int
            Max length of input encodings. Defaults to 512.

        max_target_length : int
            Max length of target encodings. Defaults to 256.

        subsets : list[str]
            Dataset subsets to return. Valid values are 'train', 'validation'
            and 'test'. Defaults to ('train', 'validation'). Half validation
            set will be used as test set, since it does not come with it, but
            the validation set is twice as big as RACE's.

        num_proc : int or None
            Number of processes to be used to transform data.

        Returns
        -------
        DatasetDict
            SQuAD's answer generation dataset.

        """

        if subsets is None:
            subsets = self._default_subsets

        squad = self._generate_squad_dataset(subsets)
        squad = self._generate_squad_ag_dataset(squad)

        dataset = {
            subset: self._map(
                squad[subset],
                self._ag_preprocessor,
                batch_size,
                max_input_length,
                max_target_length,
                num_proc
            )
            for subset in subsets
        }
        return DatasetDict(dataset)

    def get_squad_question_generation_data(
            self,
            batch_size=1024,
            max_input_length=512,
            max_target_length=256,
            subsets=None,
            num_proc=None
    ):
        """Returns question generation dataset only from SQuAD.

        Parameters
        ----------
        batch_size : int
            Batch size used to generate encodings with the tokenizer. Defaults
            to 1024.

        max_input_length : int
            Max length of input encodings. Defaults to 512.

        max_target_length : int
            Max length of target encodings. Defaults to 256.

        subsets : list[str]
            Dataset subsets to return. Valid values are 'train', 'validation'
            and 'test'. Defaults to ('train', 'validation'). Half validation
            set will be used as test set, since it does not come with it, but
            the validation set is twice as big as RACE's.

        num_proc : int or None
            Number of processes to be used to transform data.

        Returns
        -------
        DatasetDict
            SQuAD's question generation dataset.

        """

        if subsets is None:
            subsets = self._default_subsets

        squad = self._generate_squad_dataset(subsets)
        dataset = {
            subset: self._map(
                squad[subset],
                self._qg_preprocessor,
                batch_size,
                max_input_length,
                max_target_length,
                num_proc
            )
            for subset in subsets
        }
        return DatasetDict(dataset)

    def get_race_distractor_generation_data(
            self,
            batch_size=1024,
            max_input_length=512,
            max_target_length=256,
            subsets=None,
            num_proc=None
    ):
        """Returns distractor generation dataset only from RACE.

        Parameters
        ----------
        batch_size : int
            Batch size used to generate encodings with the tokenizer. Defaults
            to 1024.

        max_input_length : int
            Max length of input encodings. Defaults to 512.

        max_target_length : int
            Max length of target encodings. Defaults to 256.

        subsets : list[str]
            Dataset subsets to return. Valid values are 'train', 'validation'
            and 'test'. Defaults to ('train', 'validation').

        num_proc : int or None
            Number of processes to be used to transform data.

        Returns
        -------
        DatasetDict
            RACE's distractor generation dataset.

        """

        if subsets is None:
            subsets = self._default_subsets

        dataset = {
            subset: self._map(
                self._race[subset],
                self._dg_preprocessor,
                batch_size,
                max_input_length,
                max_target_length,
                num_proc
            )
            for subset in subsets
        }
        return DatasetDict(dataset)

    def _generate_squad_dataset(self, subsets):
        # SQuAD doesn't come with a test set, so we will use half validation
        squad = {}

        n = len(self._squad['validation'])

        if 'train' in subsets:
            squad['train'] = self._squad['train']
        if 'test' in subsets:
            select = range(n // 2, n)
            squad['test'] = self._squad['validation'].select(select)
        if 'validation' in subsets:
            select = range(n // 2)
            squad['validation'] = self._squad['validation'].select(select)

        return DatasetDict(squad)

    @staticmethod
    def _generate_squad_ag_dataset(squad):
        def concat(x):
            return np.unique(np.concatenate(list(x)))

        dataset = {}

        for subset in squad.keys():
            df = squad[subset].to_pandas()[['context', 'answers']]
            df['answers'] = df['answers'].map(lambda answer: answer['text'])
            data = Dataset.from_pandas(df.groupby('context').aggregate(concat))
            dataset[subset] = data

        return DatasetDict(dataset)

    def _map(
            self,
            dataset,
            preprocess_fn,
            batch_size,
            max_input_length,
            max_target_length,
            num_proc
    ):
        return dataset.map(
            preprocess_fn,
            remove_columns=dataset.column_names,
            num_proc=num_proc
        ).map(
            self._tokenize(max_input_length, max_target_length),
            batched=batch_size > 1,
            batch_size=batch_size,
            remove_columns=['input_text', 'target_text'],
            num_proc=num_proc
        )

    def _tokenize(self, max_input_length, max_target_length):
        def tokenize(example):
            inputs = self._tokenizer(
                example['input_text'],
                padding='longest',
                max_length=max_input_length,
                truncation=True,
                return_tensors='pt'
            )
            outputs = self._tokenizer(
                example['target_text'],
                padding='longest',
                max_length=max_target_length,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = inputs.input_ids.numpy()
            attention_mask = inputs.attention_mask.numpy()
            labels = outputs.input_ids.numpy()

            labels[labels == self._tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

        return tokenize
