import datetime

from training.arguments import parse_args
from training.train import train
from mcq.mcqg_multitask_dataset import MCQMultitaskDataset


def main():
    start = datetime.datetime.now()

    args = parse_args()
    model = train(MCQMultitaskDataset, args)

    # Save model
    model.save_pretrained(f'{args.output_dir}/model-{start:%Y%m%d%H%M}')


if __name__ == '__main__':
    main()
