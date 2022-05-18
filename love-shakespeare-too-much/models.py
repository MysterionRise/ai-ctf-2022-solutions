from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

from utils import START_SEQ, END_SEQ

MAX_LENGTH = 100


class CharLSTMCell(nn.Module):
    """
    Implement the scheme above as torch module
    """

    def __init__(self, num_tokens, padding_idx, embedding_size=100, rnn_num_units=200):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units

        self.embedding = nn.Embedding(
            num_tokens, embedding_size, padding_idx=padding_idx
        )
        self.rnn_update = nn.LSTMCell(embedding_size, rnn_num_units, bias=True)

        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, state_prev):
        """
        This method computes h_next(x, h_prev) and log P(x_next | h_next)
        We'll call it repeatedly to produce the whole sequence.

        :param state_prev: previous rnn hidden states, containing matrix
        [batch, rnn_num_units] of float32
        :param x: batch of character ids, containing vector of int64
        """
        # get vector embedding of x
        x_emb = self.embedding(x)

        state_next = self.rnn_update(x_emb, state_prev)

        # compute logits for next character probs
        logits = self.rnn_to_logits(state_next[0])

        return state_next, F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        """ return rnn state before it processes first input (aka h0) """
        hid_state = np.random.uniform(
            low=-1.0, high=1.0, size=(batch_size, self.num_units)
        )
        cell_state = np.random.uniform(
            low=-1.0, high=1.0, size=(batch_size, self.num_units)
        )
        return torch.Tensor(hid_state), torch.Tensor(cell_state)


@torch.no_grad()
def generate_sample(
        char_rnn, hid_state, token_to_id, id_to_token,
        seed_phrase=START_SEQ,  max_length=MAX_LENGTH, temperature=1.0
):
    """
    The function generates text given a phrase of length at least SEQ_LENGTH.
    :param hid_state: state of context
    :param id_to_token: map index to token
    :param token_to_id: map token to index
    :param char_rnn: character level RNN model
    :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
    :param max_length: maximum output length, including seed_phrase
    :param temperature: coefficient for sampling.  higher temperature produces more
    chaotic outputs, smaller temperature converges to the single most likely output
    """

    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64)

    # feed the seed phrase, if any
    for i in range(len(seed_phrase) - 1):
        hid_state, _ = char_rnn(x_sequence[:, i], hid_state)

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = char_rnn(x_sequence[:, -1], hid_state)
        p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0]

        # sample next token and push it back into x_sequence
        next_ix = np.random.choice(len(token_to_id), p=p_next)
        if next_ix == token_to_id[END_SEQ]:
            break
        else:
            next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
            x_sequence = torch.cat([x_sequence, next_ix], dim=1)

    return ''.join([id_to_token[ix] for ix in x_sequence.data.numpy()[0]])