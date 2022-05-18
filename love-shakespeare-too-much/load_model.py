from models import CharLSTMCell, generate_sample
from utils import PAD

import json
import torch

PATH_LSTM = 'poetry_lstm_128_128.w'


def load_model(path, params, ModelClass):
    char_model = ModelClass(**params)
    char_model.load_state_dict(torch.load(path))
    char_model.eval()
    return char_model


with open('token_to_id.json', 'rb') as f:
    token_to_id = json.load(f)

id_to_token = dict((idx, t) for t, idx in token_to_id.items())


fit_params = {
    'num_tokens': len(token_to_id),
    'padding_idx': token_to_id[PAD],
    'embedding_size': 128,
    'rnn_num_units': 128
}

char_lstm = load_model(PATH_LSTM, fit_params, CharLSTMCell)


def get_raw():
    hid_state = char_lstm.initial_state(batch_size=1)
    sample = generate_sample(
        char_lstm, hid_state, token_to_id, id_to_token, temperature=0.5
    )
    return sample
