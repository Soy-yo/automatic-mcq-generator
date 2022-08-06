import abc

from datasets import load_dataset


class AbstractPreprocessor(abc.ABC):

    @abc.abstractmethod
    def get_preprocessed(
            self,
            batch_size,
            shuffle_train=False,
            shuffle_test=False,
            select_train=None,
            select_test=None,
            subset=None,
            num_proc=None
    ):
        pass


class DatasetPreprocessor(AbstractPreprocessor):

    def __init__(
            self,
            tokenizer,
            max_input_length=None,
            max_target_length=None
    ):
        self._tokenizer = tokenizer
        self._max_input_length = max_input_length
        self._max_target_length = max_target_length
        self._dataset = self._get_dataset()

    @abc.abstractmethod
    def _get_dataset(self):
        pass

    @abc.abstractmethod
    def _preprocess(self, dataset, batch_size, num_proc):
        pass

    def get_preprocessed(
            self,
            batch_size,
            shuffle_train=False,
            shuffle_test=False,
            select_train=None,
            select_test=None,
            subset=None,
            num_proc=None
    ):
        assert subset in (None, 'train', 'test')

        train = self._preprocess_main(
            self._dataset['train'],
            batch_size,
            shuffle_train,
            select_train,
            num_proc
        ) if subset in (None, 'train') else None
        test = self._preprocess_main(
            self._dataset['validation'],
            batch_size,
            shuffle_test,
            select_test,
            num_proc
        ) if subset in (None, 'test') else None

        if subset is None:
            return train, test

        return {'train': train, 'test': test}[subset]

    def _preprocess_main(
            self,
            dataset,
            batch_size,
            shuffle,
            select,
            num_proc
    ):
        if shuffle is True or shuffle == 'before':
            dataset = dataset.shuffle()

        if select is not None:
            if isinstance(select, float):
                select = int(len(dataset) * select)
            if isinstance(select, int):
                select = range(select)
            dataset = dataset.select(select)

        dataset = self._preprocess(dataset, batch_size, num_proc)

        if shuffle == 'after':
            dataset = dataset.shuffle()

        return dataset

    def _tokenize(self, example):
        inputs = self._tokenizer(
            example['input_text'],
            padding='longest',
            max_length=self._max_input_length,
            truncation=True,
            return_tensors='pt'
        )
        outputs = self._tokenizer(
            example['target_text'],
            padding='longest',
            max_length=self._max_target_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs.input_ids.numpy()
        attention_mask = inputs.attention_mask.numpy()
        labels = outputs.input_ids.numpy()

        # Replace padding token id's of the labels by -100 (ignore token)
        labels[labels == self._tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class BaseDatasetPreprocessor(DatasetPreprocessor):

    _dataset_args = ()
    _dataset_kwargs = {}

    @property
    @abc.abstractmethod
    def _dataset_name(self):
        pass

    def _get_dataset(self):
        return load_dataset(
            self._dataset_name,
            *self._dataset_args,
            **self._dataset_kwargs
        )
