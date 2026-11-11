import itertools

import pytest

from configurations import N_CONFIGS, SHAPE, caption, decode, encode


def test_roundtrip_and_bijection():
    configs = list(itertools.product(*(range(n) for n in SHAPE)))
    indices = [encode(c) for c in configs]
    assert indices == list(range(N_CONFIGS))  # row-major order, no gap, no duplicate
    assert all(decode(i) == c for i, c in zip(indices, configs, strict=True))


def test_caption():
    assert caption((2, 0, 0, 0, 2)) == "a small red cone, camera at the top, on a salmon background"


@pytest.mark.parametrize("config", [(0, 0, 0), (6, 0, 0, 0, 0), (0, 0, 0, 0, -1)])
def test_encode_rejects_bad_config(config):
    with pytest.raises(ValueError):
        encode(config)


@pytest.mark.parametrize("index", [-1, N_CONFIGS])
def test_decode_rejects_bad_index(index):
    with pytest.raises(ValueError):
        decode(index)
