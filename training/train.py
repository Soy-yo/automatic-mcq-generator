import datetime

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer
)


def train(dataset_class, args):
    """Fine-tunes a pretrained T% model with a squad dataset.

    Parameters
    ----------
    dataset_class : type[mcq.dataset_preprocessor.AbstractPreprocessor]
        Dataset subclass for specific task.
    args : training.arguments.CommandLineArguments
        Command line arguments used for training.

    Returns
    -------
    transformers.modeling_utils.PreTrainedModel
        Model specified by args trained on specific task.

    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Load and preprocess SQuAD dataset
    dataset = dataset_class(
        tokenizer,
        args.max_input_length,
        args.max_target_length
    )
    train_dataset = dataset.get_preprocessed(
        args.batch_size,
        subset='train',
        num_proc=args.num_proc
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model,
        pad_to_multiple_of=8
    )

    # Generate training arguments
    training_args = args.get_training_args()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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
