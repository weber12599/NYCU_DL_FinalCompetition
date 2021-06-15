import torch


class Dataset():
    def __init__(
        self,
        dataset_dir: str,
        split: list
    ) -> None:
        super().__init__()
        return


def main():
    dataset = Dataset(
        dataset_dir='./dataset',
        split=['train', 'test']
    )
    return


if __name__ == '__main__':
    main()
