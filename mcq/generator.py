from typing import List

import torch
from dataclasses import dataclass, field, asdict

from .data import (
    SQuADAnswerGenerationPreprocessor,
    SQuADQuestionGenerationPreprocessor,
    RACEDistractorGenerationPreprocessor
)


DEFAULT_ANSWER_TOKENIZER = 't5-small'
DEFAULT_ANSWER_MODEL = 't5-small'
DEFAULT_QUESTION_TOKENIZER = 't5-small'
DEFAULT_QUESTION_MODEL = 't5-small'
DEFAULT_DISTRACTOR_TOKENIZER = 't5-small'
DEFAULT_DISTRACTOR_MODEL = 't5-small'

DEFAULT_MAX_CONTEXT_TOKENS = 1024
DEFAULT_MAX_ANSWER_TOKENS = 30
DEFAULT_MAX_QUESTION_TOKENS = 50
DEFAULT_MAX_DISTRACTOR_TOKENS = 30

DEFAULT_MAX_TOTAL_ANSWER_TOKENS = 256
DEFAULT_MAX_TOTAL_DISTRACTOR_TOKENS = 256

DEFAULT_BATCH_SIZE = 1


def get_models(tokenizer, model):
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if isinstance(model, str):
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model)

    return tokenizer, model


def flatten(xss):
    return (x for xs in xss for x in xs)


@dataclass
class DocumentMCQ:
    question: str = field()
    answer: str = field()
    distractors: List[str] = field()

    def dict(self):
        return asdict(self)


@dataclass
class MCQResult:
    document: str = field()
    mcqs: List[DocumentMCQ] = field()

    def dict(self):
        return asdict(self)


class AnswerGenerator:

    def __init__(
            self,
            tokenizer=DEFAULT_ANSWER_TOKENIZER,
            model=DEFAULT_ANSWER_MODEL,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            max_tokens_per_answer=DEFAULT_MAX_ANSWER_TOKENS,
            max_total_answer_tokens=DEFAULT_MAX_TOTAL_ANSWER_TOKENS,
            preprocessor=None
    ):
        self._tokenizer, self._model = get_models(tokenizer, model)

        if preprocessor is None:
            preprocessor = SQuADAnswerGenerationPreprocessor(
                self._tokenizer,
                with_target=False
            )

        self.max_context_tokens = max_context_tokens
        self.max_tokens_per_answer = max_tokens_per_answer
        self.max_total_answer_tokens = max_total_answer_tokens
        self._preprocessor = preprocessor

        self.contexts = None
        self.context_tokens = None

    def fit(self, documents):
        self.contexts = [self._preprocessor(context=d) for d in documents]

        self.context_tokens = [
            self._tokenizer(
                [context],
                max_length=self.max_context_tokens,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            for context in self.contexts
        ]

        return self

    def generate(self, answers_per_document=3, **kwargs):
        def answers_range(n):
            if n >= 0:
                return range(n)

            def infinite_range():
                it = 0
                while True:
                    yield it
                    it += 1

            return infinite_range()

        kwargs.setdefault('max_new_tokens', self.max_tokens_per_answer)
        answers = []

        for context in self.context_tokens:
            context['decoder_input_ids'] = torch.tensor([[
                self._tokenizer.pad_token_id
            ]])

            context_answer_tokens = []

            k = 0

            for _ in answers_range(answers_per_document):
                output = self._model.generate(
                    **context,
                    forced_eos_token_id=self._tokenizer.eos_token_id,
                    **kwargs
                )

                # Add previous generations to decoder input
                context['decoder_input_ids'] = output

                context_answer_tokens.append(output[0, k:])

                # Stop if maximum size will be reached
                k = len(output[0])
                next_max = k + self.max_tokens_per_answer
                if next_max > self.max_total_answer_tokens:
                    break

            # Extract all generated answers
            if context_answer_tokens:
                context_answers = self._tokenizer.batch_decode(
                    context_answer_tokens,
                    skip_special_tokens=True
                )
            else:
                context_answers = []

            answers.append(context_answers)

        return answers


class QuestionGenerator:

    def __init__(
            self,
            tokenizer=DEFAULT_QUESTION_TOKENIZER,
            model=DEFAULT_QUESTION_MODEL,
            batch_size=DEFAULT_BATCH_SIZE,
            max_input_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            max_question_tokens=DEFAULT_MAX_QUESTION_TOKENS,
            preprocessor=None
    ):
        self._tokenizer, self._model = get_models(tokenizer, model)

        if preprocessor is None:
            preprocessor = SQuADQuestionGenerationPreprocessor(
                self._tokenizer,
                with_target=False
            )

        self.batch_size = batch_size

        self.max_input_tokens = max_input_tokens
        self.max_question_tokens = max_question_tokens
        self._preprocessor = preprocessor

        self.inputs = None
        self.input_tokens = None

    def fit(self, documents, answers):
        self.inputs = [
            self._preprocessor(context=d, answer=a)
            for d, a in zip(documents, answers)
        ]

        self.input_tokens = [
            self._tokenizer(
                self.inputs[i:i+self.batch_size],
                max_length=self.max_input_tokens,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            for i in range(0, len(self.inputs), self.batch_size)
        ]

        return self

    def generate(self, **kwargs):
        kwargs.setdefault('max_new_tokens', self.max_question_tokens)
        questions = []
        for batch in self.input_tokens:
            output = self._model.generate(**batch, **kwargs)
            batch_questions = self._tokenizer.batch_decode(
                output,
                skip_special_tokens=True
            )
            questions.extend(batch_questions)

        return questions


class DistractorGenerator:

    def __init__(
            self,
            tokenizer=DEFAULT_DISTRACTOR_TOKENIZER,
            model=DEFAULT_DISTRACTOR_MODEL,
            max_input_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            max_tokens_per_distractor=DEFAULT_MAX_DISTRACTOR_TOKENS,
            max_total_distractor_tokens=DEFAULT_MAX_TOTAL_DISTRACTOR_TOKENS,
            preprocessor=None
    ):
        self._tokenizer, self._model = get_models(tokenizer, model)

        if preprocessor is None:
            preprocessor = RACEDistractorGenerationPreprocessor(
                self._tokenizer,
                with_target=False
            )

        self.max_input_tokens = max_input_tokens
        self.max_tokens_per_distractor = max_tokens_per_distractor
        self.max_total_distractor_tokens = max_total_distractor_tokens
        self._preprocessor = preprocessor

        self.inputs = None
        self.input_tokens = None

    def fit(self, documents, answers, questions):
        self.inputs = [
            self._preprocessor(question=q, answer=a, context=d)
            for d, a, q in zip(documents, answers, questions)
        ]

        self.input_tokens = [
            self._tokenizer(
                [inp],
                max_length=self.max_input_tokens,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            for inp in self.inputs
        ]

        return self

    def generate(self, distractors_per_document=3, **kwargs):
        def distractors_range(n):
            if n >= 0:
                return range(n)

            def infinite_range():
                it = 0
                while True:
                    yield it
                    it += 1

            return infinite_range()

        kwargs.setdefault('max_new_tokens', self.max_tokens_per_distractor)

        distractors = []

        # TODO repeat generation if there are some repetition

        for inp in self.input_tokens:
            inp['decoder_input_ids'] = torch.tensor([[
                self._tokenizer.pad_token_id
            ]])

            distractor_tokens = []

            k = 0

            for _ in distractors_range(distractors_per_document):
                output = self._model.generate(
                    **inp,
                    forced_eos_token_id=self._tokenizer.eos_token_id,
                    **kwargs
                )

                # Add previous generations to decoder input
                inp['decoder_input_ids'] = output

                distractor_tokens.append(output[0, k:])

                # Stop if maximum size will be reached
                k = len(output[0])
                next_max = k + self.max_tokens_per_distractor
                if next_max > self.max_total_distractor_tokens:
                    break

            # Extract all generated answers
            if distractor_tokens:
                current_distactors = self._tokenizer.batch_decode(
                    distractor_tokens,
                    skip_special_tokens=True
                )
            else:
                current_distactors = []

            distractors.append(current_distactors)

        return distractors


class MultipleChoiceQuestionGenerator:

    def __init__(self, agen, qgen, dgen):
        self._answer_generator = agen
        self._question_generator = qgen
        self._distractor_generator = dgen

        self.documents = None

    def fit(self, documents):
        self.documents = documents
        return self

    def generate(
            self,
            questions_per_document=3,
            *,
            distractor_generation_merge_documents=0,
            ag_kwargs=None,
            qg_kwargs=None,
            dg_kwargs=None,
            **kwargs
    ):
        if ag_kwargs is None:
            ag_kwargs = {}
        if qg_kwargs is None:
            qg_kwargs = {}
        if dg_kwargs is None:
            dg_kwargs = {}

        # kwargs must be shared by all generate methods
        ag_kwargs = {**kwargs, **ag_kwargs}
        qg_kwargs = {**kwargs, **qg_kwargs}
        dg_kwargs = {**kwargs, **dg_kwargs}

        self._answer_generator.fit(self.documents)
        answers = self._answer_generator.generate(
            questions_per_document,
            **ag_kwargs
        )

        # Repeat every document once per answer
        documents, flat_answers = self._match_documents_and_answers(answers)

        self._question_generator.fit(documents, flat_answers)
        questions = self._question_generator.generate(**qg_kwargs)

        if (
                distractor_generation_merge_documents == 'all' or
                distractor_generation_merge_documents >= len(self.documents)
        ):
            def repeat_doc(doc):
                while True:
                    yield doc

            all_docs = repeat_doc('\n'.join(self.documents))
            self._distractor_generator.fit(all_docs, flat_answers, questions)
        elif distractor_generation_merge_documents == 0:
            self._distractor_generator.fit(documents, flat_answers, questions)
        else:
            merged_docs = self._merge_docs(
                answers,
                distractor_generation_merge_documents
            )
            self._distractor_generator.fit(merged_docs, flat_answers, questions)

        distractors = self._distractor_generator.generate(**dg_kwargs)

        return self._prepare_results(questions, answers, distractors)

    def _match_documents_and_answers(self, answers):
        documents = list(flatten(
            [doc] * len(ans)
            for doc, ans in zip(self.documents, answers)
        ))
        answers = list(flatten(answers))
        return documents, answers

    def _merge_docs(self, answers, k):
        result = []
        n = len(self.documents)
        # We don't count the central doc
        k -= 1
        before = k // 2
        after = k - before
        for i, (doc, doc_answers) in enumerate(zip(self.documents, answers)):
            start = i - before
            end = i + after + 1
            if start < 0:
                start = 0
                end += -start
            elif end > n:
                start -= end - n
                end = n

            merged_doc = '\n'.join(self.documents[start:end])
            result.extend([merged_doc] * len(doc_answers))

        return result

    def _prepare_results(self, questions, answers, distractors):
        results = []
        k = 0
        # Questions and distractors are flattened
        # We need to match them with their answers
        for doc, doc_answers in zip(self.documents, answers):
            mcqs = []
            for answer in doc_answers:
                mcqs.append(DocumentMCQ(questions[k], answer, distractors[k]))
                k += 1

            results.append(MCQResult(doc, mcqs))

        return results
