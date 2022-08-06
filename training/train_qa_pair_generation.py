import datetime

from training.arguments import parse_args
from training.train import train
from mcq.mcqg_multitask_dataset import QuestionAnswerPairGenerationDataset


def main():
    start = datetime.datetime.now()

    args = parse_args()
    model = train(QuestionAnswerPairGenerationDataset, args)

    # Save model
    model.save_pretrained(f'{args.output_dir}/model-{start:%Y%m%d%H%M}')


if __name__ == '__main__':
    main()
