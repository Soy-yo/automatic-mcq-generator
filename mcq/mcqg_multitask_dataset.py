import abc

from datasets import concatenate_datasets

from .dataset_preprocessor import AbstractPreprocessor
from .squad import SQuADAnswerGenerationDataset, SQuADQuestionGenerationDataset
from .race import RACEDistractorGenerationDataset


class AbstractMultitaskDataset(AbstractPreprocessor):

    def __init__(
            self,
            tokenizer,
            max_input_length=None,
            max_target_length=None
    ):
        dataset_classes = self._get_dataset_classes()
        n = len(dataset_classes)

        if max_input_length is None:
            max_input_length = (None,) * n
        elif isinstance(max_input_length, int):
            max_input_length = (max_input_length,) * n
        if max_target_length is None:
            max_target_length = (None,) * n
        elif isinstance(max_target_length, int):
            max_target_length = (max_target_length,) * n

        self._datasets = [
            cls(tokenizer, max_inp, max_tgt)
            for cls, max_inp, max_tgt
            in zip(dataset_classes, max_input_length, max_target_length)
        ]
        self._n = n

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
        if select_train is not None:
            select_train = select_train // self._n
        if select_test is not None:
            select_test = select_test // self._n

        data = [
            dataset.get_preprocessed(
                batch_size,
                shuffle_train,
                shuffle_test,
                select_train,
                select_test,
                subset,
                num_proc
            ) for dataset in self._datasets
        ]

        if subset is not None:
            data = [(d,) for d in data]

        multitask_dataset = tuple(
            concatenate_datasets(list(split)) for split in zip(*data)
        )

        if subset is not None:
            return multitask_dataset[0]

        return multitask_dataset

    @abc.abstractmethod
    def _get_dataset_classes(self):
        pass


class QuestionAnswerPairGenerationDataset(AbstractMultitaskDataset):

    def _get_dataset_classes(self):
        return SQuADAnswerGenerationDataset, SQuADQuestionGenerationDataset


class MCQMultitaskDataset(AbstractMultitaskDataset):

    def _get_dataset_classes(self):
        return (
            SQuADAnswerGenerationDataset,
            SQuADQuestionGenerationDataset,
            RACEDistractorGenerationDataset
        )
