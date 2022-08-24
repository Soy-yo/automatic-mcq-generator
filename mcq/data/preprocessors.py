import abc


class ExamplePreprocessor(abc.ABC):
    """Preprocesses one example to the expected format."""

    def __init__(self, tokenizer, prefixes=(), with_target=True):
        """

        Parameters
        ----------
        tokenizer : transformers.T5Tokenizer
            Tokenizer that will be used to encode the texts.

        prefixes : list[str]
            Custom task prefixes.

        with_target : bool
            Whether to return target test as well if possible. Useful for
            training. Defaults to True.
        """
        self._tokenizer = tokenizer
        self._prefixes = prefixes
        self._with_target = with_target

    def __call__(self, example=None, **kwargs):
        """Execute preprocessing.

        Parameters
        ----------
        example : dict
            Dict-like object with keys accepted by generation methods. Usually
            used inside a dataset mapping function.

        **kwargs
            Same as example, but as keyword arguments. Usually used directly.

        Returns
        -------
        dict[str, str] or str
            Dict with keys 'input_text' and 'target_text' if with_target is
            True. Otherwise, the input text as a plain string.

        """
        if example is not None:
            kwargs.update(example)

        input_text = self._get_input_text(**kwargs)
        if self._with_target:
            target_text = self._get_target_text(**kwargs)
            return {
                'input_text': input_text,
                'target_text': target_text
            }

        return input_text

    @abc.abstractmethod
    def _get_input_text(self, **kwargs):
        pass

    @abc.abstractmethod
    def _get_target_text(self, **kwargs):
        pass


class SQuADAnswerGenerationPreprocessor(ExamplePreprocessor):
    """Preprocesses SQuAD-like examples for answer generation.

    Accepted kwargs are context (str) and answers (list[str], target).
    """

    _default_prefixes = ('generate answers',)

    def __init__(self, tokenizer, prefixes=_default_prefixes, with_target=True):
        super(SQuADAnswerGenerationPreprocessor, self).__init__(
            tokenizer,
            prefixes,
            with_target
        )

    def _get_input_text(self, *, context, **kwargs):
        return f'{self._prefixes[0]}: {context}'

    def _get_target_text(self, *, answers, **kwargs):
        if isinstance(answers, dict):
            answers = answers['text']
        return self._tokenizer.eos_token.join(answers)


class SQuADQuestionGenerationPreprocessor(ExamplePreprocessor):
    """Preprocesses SQuAD-like examples for question generation.

    Accepted kwargs are context (str), answer (str) and question (str, target).
    """

    _default_prefixes = ('generate question', 'context')

    def __init__(self, tokenizer, prefixes=_default_prefixes, with_target=True):
        super(SQuADQuestionGenerationPreprocessor, self).__init__(
            tokenizer,
            prefixes,
            with_target
        )

    def _get_input_text(self, *, context, answers=None, answer=None, **kwargs):
        if answer is None:
            if answers is None:
                raise ValueError('Set either answer or answers')
            # SQuAD answers is a dict
            answer = answers['text'][0]

        return f'{self._prefixes[0]}: {answer} {self._prefixes[1]}: {context}'

    def _get_target_text(self, *, question, **kwargs):
        return question


class RACEDistractorGenerationPreprocessor(ExamplePreprocessor):
    """Preprocesses RACE-like examples for distractor generation.

    Accepted kwargs are context or article (str), answer (str), question (str)
    and options (list[str], target).
    """

    _default_prefixes = ('generate distractors', 'answer', 'context')
    _answer_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def __init__(self, tokenizer, prefixes=_default_prefixes, with_target=True):
        super(RACEDistractorGenerationPreprocessor, self).__init__(
            tokenizer,
            prefixes,
            with_target
        )

    def _get_input_text(
            self, *,
            question,
            answer,
            article=None,
            options=None,
            context=None,
            **kwargs
    ):
        if article is None:
            if context is None:
                raise ValueError('Set either article or context')
            # Allow context as well to match other preprocessor interfaces
            # article is here since RACE's context is named article
            article = context

        if options is not None:
            # Let's assume if we receive options the answer is just the letter
            answer_index = self._answer_indices[answer]
            answer = options[answer_index]

        return (
            f'{self._prefixes[0]}: {question} '
            f'{self._prefixes[1]}: {answer} '
            f'{self._prefixes[2]}: {article}'
        )

    def _get_target_text(self, *, options, answer, **kwargs):
        if isinstance(answer, int):
            answer_index = answer
        else:
            answer_index = self._answer_indices[answer]
        targets = [
            option
            for i, option in enumerate(options)
            if i != answer_index
        ]
        return self._tokenizer.eos_token.join(targets)
