from argparse import ArgumentParser
from dataclasses import dataclass, field

from transformers import TrainingArguments

MODEL_SIZES = ['small', 'base', 'large', 't3b', '11b']
DEFAULT_OUTPUT_DIR = 'trains'
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_TARGET_LENGTH = 128
# https://huggingface.co/docs/transformers/model_doc/t5
DEFAULT_LEARNING_RATE = 3e-4


@dataclass
class CommandLineArguments:
    model_name: str = field(default=f't5-{MODEL_SIZES[0]}')
    output_dir: str = field(default=DEFAULT_OUTPUT_DIR)
    max_input_length: int = field(default=DEFAULT_MAX_INPUT_LENGTH)
    max_target_length: int = field(default=DEFAULT_MAX_TARGET_LENGTH)
    batch_size: int = field(default=DEFAULT_BATCH_SIZE)
    learning_rate: float = field(default=DEFAULT_LEARNING_RATE)
    epochs: float = field(default=None)
    n_examples: int = field(default=None)
    save_steps: int = field(default=None)
    num_proc: int = field(default=None)

    def get_training_args(self):
        kwargs = {
            'output_dir': self.output_dir,
            'per_device_train_batch_size': self.batch_size,
            'optim': 'adafactor',
            'learning_rate': self.learning_rate
        }
        if self.epochs is not None:
            kwargs['num_train_epochs'] = self.epochs
        if self.n_examples is not None:
            kwargs['max_steps'] = self.n_examples // self.batch_size
        if self.save_steps is not None:
            kwargs['save_steps'] = self.save_steps // self.batch_size
        return TrainingArguments(**kwargs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-s', '--size',
        type=str,
        choices=MODEL_SIZES,
        default=MODEL_SIZES[1],
        help='Size of the T5 model to be trained'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to save trained model at'
    )
    parser.add_argument(
        '--max-input-tensor-length',
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
        help='Maximum length of input tensors to the model'
    )
    parser.add_argument(
        '--max-output-tensor-length',
        type=int,
        default=DEFAULT_MAX_TARGET_LENGTH,
        help='Maximum length of output tensors of the model'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '-l', '--lr', '--learning-rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate for training'
    )
    parser.add_argument(
        '-n', '--epochs',
        type=float,
        help='Number of training epochs'
    )
    parser.add_argument(
        '-N', '--examples',
        type=int,
        help='Number of training examples to be shown to the model '
             '(overrides epochs)'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        help='Number of training steps between each checkpoint'
    )
    parser.add_argument(
        '-p', '--proc',
        type=int,
        help='Number of processors to be used in some operations'
    )
    args = parser.parse_args()

    kwargs = {
        'model_name': f't5-{args.size}',
        'output_dir': args.output,
        'max_input_length': args.max_input_tensor_length,
        'max_target_length': args.max_output_tensor_length,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }

    if args.epochs is not None:
        kwargs['epochs'] = args.epochs

    if args.examples is not None:
        kwargs['n_examples'] = args.examples

    if args.save_steps is not None:
        kwargs['save_steps'] = args.save_steps

    if args.proc is not None:
        kwargs['num_proc'] = args.proc

    return CommandLineArguments(**kwargs)
