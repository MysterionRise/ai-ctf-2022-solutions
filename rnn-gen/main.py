import argparse

import torch

from encryptor import Encryptor
from generator import REGISTER_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN based generator pseudo-random numbers"
    )
    parser.add_argument(
        "--input_file",
        dest="input_file",
        type=str,
        help="file for encryption/decryption",
    )

    parser.add_argument(
        "--output_file", dest="output_file", type=str, help="file for result"
    )

    parser.add_argument(
        "--x_0_init", dest="x_0_init", type=str, help="initial state for first register"
    )

    parser.add_argument(
        "--x_1_init",
        dest="x_1_init",
        type=str,
        help="initial state for second register",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    x_0_init = torch.Tensor([eval(args.x_0_init)])
    x_1_init = torch.Tensor([eval(args.x_1_init)])

    assert x_0_init.shape == (1, REGISTER_SIZE), "not correct init state size for x0"
    assert x_1_init.shape == (1, REGISTER_SIZE), "not correct init state size for x1"

    enc = Encryptor(x_0_init, x_1_init)
    enc.apply(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
