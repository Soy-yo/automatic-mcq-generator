import os
import json
from argparse import ArgumentParser
from dataclasses import dataclass, field, fields
import datetime

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

from mcq.data import MultitaskMCQDataset


@dataclass(frozen=True)
class SpecificPreprocessConfig:
    max_input_length: int = field(default=None)
    max_target_length: int = field(default=None)


@dataclass(frozen=True)
class PreprocessConfig:
    batch_size: int = field(default=512)
    use_validation: bool = field(default=True)
    num_proc: int = field(default=None)
    ag: SpecificPreprocessConfig = field(default=SpecificPreprocessConfig())
    qg: SpecificPreprocessConfig = field(default=SpecificPreprocessConfig())
    dg: SpecificPreprocessConfig = field(default=SpecificPreprocessConfig())


@dataclass(frozen=True)
class Config:
    model_name: str = field(default='t5-small')
    save_to: str = field(default='.')
    preprocess_config: PreprocessConfig = field(default=PreprocessConfig())
    training_args: TrainingArguments = field(default=TrainingArguments('.'))


def parse_config(filename):
    """Returns a configuration object from the given json file.

    Parameters
    ----------
    filename : str
        Name of the config file.

    Returns
    -------
    Config
        Parsed config.

    """
    with open(filename) as f:
        contents = json.load(f)

    preprocess_config_content = contents.pop('preprocess_config', {})
    ag_specific_config_content = preprocess_config_content.pop('ag', {})
    qg_specific_config_content = preprocess_config_content.pop('qg', {})
    dg_specific_config_content = preprocess_config_content.pop('dg', {})

    # Fields can be set globally in preprocess_config
    for attr in fields(SpecificPreprocessConfig):
        attr_name = attr.name
        if attr_name in preprocess_config_content:
            value = preprocess_config_content.pop(attr_name)
            ag_specific_config_content.setdefault(attr_name, value)
            qg_specific_config_content.setdefault(attr_name, value)
            dg_specific_config_content.setdefault(attr_name, value)

    ag_specific_config = SpecificPreprocessConfig(**ag_specific_config_content)
    qg_specific_config = SpecificPreprocessConfig(**qg_specific_config_content)
    dg_specific_config = SpecificPreprocessConfig(**dg_specific_config_content)

    preprocess_config = PreprocessConfig(
        **preprocess_config_content,
        ag=ag_specific_config,
        qg=qg_specific_config,
        dg=dg_specific_config
    )

    training_args_content = contents.pop('training_args', {})
    training_args_content.setdefault('output_dir', '.')
    training_args = TrainingArguments(**training_args_content)

    return Config(
        **contents,
        preprocess_config=preprocess_config,
        training_args=training_args
    )


def train(config):
    """Fine-tunes a pretrained T5 model for Multiple Choice Question Generation.

    Parameters
    ----------
    config : Config
        Configuration to be used.

    Returns
    -------
    transformers.modeling_utils.PreTrainedModel
        Model specified by args trained on specific task.

    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)

    # Load and preprocess SQuAD dataset
    dataset = MultitaskMCQDataset(tokenizer)

    subsets = (
        ('train', 'validation')
        if config.preprocess_config.use_validation else ('train',)
    )

    dataset = dataset.get_multitask(
        batch_size=config.preprocess_config.batch_size,
        max_input_length={
            'ag': config.preprocess_config.ag.max_input_length,
            'qg': config.preprocess_config.qg.max_input_length,
            'dg': config.preprocess_config.dg.max_input_length,
        },
        max_target_length={
            'ag': config.preprocess_config.ag.max_target_length,
            'qg': config.preprocess_config.qg.max_target_length,
            'dg': config.preprocess_config.dg.max_target_length,
        },
        subsets=subsets,
        num_proc=config.preprocess_config.num_proc
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        pad_to_multiple_of=8
    )

    # Generate training arguments
    training_args = config.training_args

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator
    )

    # Train
    start = datetime.datetime.now()
    print(f'Starting training at {start:%Y-%m-%d %H:%M:%S}')

    trainer.train()

    end = datetime.datetime.now()
    print(f'Finished training at {end:%Y-%m-%d %H:%M:%S}')
    print(f'Approximate time spent: {end - start}')

    return model


def main():
    start = datetime.datetime.now()

    parser = ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Config filename location (json).'
    )

    args = parser.parse_args()

    if args.config is None:
        config = Config()
    else:
        config = parse_config(args.config)

    model = train(config)

    # Save model
    model.save_pretrained(os.path.join(config.save_to, f'{start:%Y%m%d%H%M%S}'))


if __name__ == '__main__':
    main()
