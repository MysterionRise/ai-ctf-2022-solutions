from pathlib import Path

from tqdm import tqdm

from generator import S0, S1, Generator


def _write_file(path: Path, bytes_for_write) -> None:
    """
    File writer
    :param path: path for writing file
    :param bytes_for_write:
    :return:
    """
    with open(path, "wb") as file:
        file.write(bytes_for_write)


def _read_file(path: Path) -> bytes:
    """
    File reader
    :param path: path for reading
    :return: file content
    """
    with open(path, "rb") as file:
        file_bytes = file.read()

    return file_bytes


class Encryptor(Generator):
    """
    Encryprion-decryption class
    """

    def __init__(self, x_0_init, x_1_init, array_0=S0, array_1=S1):
        """

        :param x_0_init: initial state for first register
        :param x_1_init: initial state for second register
        :param array_0: linear feedback parameter for first register
        :param array_1: linear feedback parameter for second register
        """
        super().__init__(x_0_init, x_1_init, array_0, array_1)

    def apply(self, path: Path, out_path: Path) -> None:
        """
        Apply generator to file
        :param path: file path for encryption/decryption
        :param out_path: file path for result
        :return:
        """
        file_bytes = _read_file(path)

        applied_bytes = bytes(
            (256 + self.generate_byte() - byte) % 256 for byte in tqdm(file_bytes)
        )

        _write_file(out_path, applied_bytes)
