from .dataset_preprocessor import BaseDatasetPreprocessor


class RACEDistractorGenerationDataset(BaseDatasetPreprocessor):

    _dataset_name = 'race'
    _dataset_args = ('all',)
    _answer_index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    _remove_columns = ['example_id', 'article', 'question', 'answer', 'options']

    def _preprocess(self, dataset, batch_size, num_proc):
        return dataset.map(self._preprocess_data).map(
            self._tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=self._remove_columns
        )

    def _preprocess_data(self, example):
        answer_index = self._answer_index_map[example['answer']]
        targets = [
            option
            for i, option in enumerate(example['options'])
            if i != answer_index
        ]
        return {
            'input_text': f"generate distractors: {example['question']} "
                          f"answer: {example['options'][answer_index]} "
                          f"context: {example['article']}",
            'target_text': self._tokenizer.eos_token.join(targets)
        }
