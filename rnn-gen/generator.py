from collections import OrderedDict

import torch

_EXPECTED_VALUES = [0.0, 1.0]

REGISTER_SIZE = 16

S0 = torch.roll(torch.eye(REGISTER_SIZE), -1, 0)
non_zero_idx_0 = [0, 1, 3, 5]
S0[-1, non_zero_idx_0] = 1
S0 = S0 / 2

S1 = torch.roll(torch.eye(REGISTER_SIZE), -1, 0)
non_zero_idx_1 = [0, 1, 3, 12]
S1[-1, non_zero_idx_1] = 1
S1 = S1 / 2


def _init_cell(array: torch.Tensor) -> torch.nn.Module:
    cell = torch.nn.RNNCell(
        REGISTER_SIZE, REGISTER_SIZE, nonlinearity="relu", bias=False
    )
    cell.load_state_dict(
        OrderedDict(
            [
                ("weight_ih", array),
                ("weight_hh", array),
            ]
        )
    )
    return cell


class Register:
    """
    Shift register with linear feedback with the function of complication
    """

    def __init__(self, x_init: torch.Tensor, array: torch.Tensor):
        """

        :param x_init: initial state for register
        :param array: linear feedback parameter
        """
        x_unique_values = set(x_init.unique().cpu().numpy())
        assert x_unique_values.issubset(_EXPECTED_VALUES), "expect only binary values"

        self.x_init = x_init

        self._x = torch.clone(x_init)
        self.cell = _init_cell(array)

    def _f(self) -> int:
        """
        Complication function for current register state
        :return: binary value 0-1
        """
        x_0 = self._x.reshape(-1, 2)[:, 0]
        x_1 = self._x.reshape(-1, 2)[:, 1]
        return (x_0 * x_1).sum().item() % 2

    def generate_bit(self) -> int:
        """
        Get next bit from register
        :return: binary value 0-1
        """
        self._x = self.cell(self._x, self._x) % 2
        return self._f()


class Generator:
    """
    Generator of pseudo-random values
    """

    def __init__(
        self,
        x_0_init: torch.Tensor,
        x_1_init: torch.Tensor,
        array_0: torch.Tensor,
        array_1: torch.Tensor,
    ):
        """

        :param x_0_init: initial state for first register
        :param x_1_init: initial state for second register
        :param array_0: linear feedback parameter for first register
        :param array_1: linear feedback parameter for second register
        """
        self.register_0 = Register(x_0_init, array_0)
        self.register_1 = Register(x_1_init, array_1)

    def generate_byte(self) -> int:
        """
        Get next byte from generator
        :return: int value from 0 to 255
        """
        bit_list = []

        for _ in range(8):
            bit0 = self.register_0.generate_bit()
            bit1 = self.register_1.generate_bit()
            bit_list.append((bit0 + bit1) % 2)

        byte = "".join(map(lambda x: str(int(x)), bit_list))

        return int(byte, base=2)
