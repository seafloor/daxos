from daxos import utils
import numpy as np
import pytest


def test_parse_bool():
    assert not utils.parse_bool('False')
    assert not utils.parse_bool('false')

    assert utils.parse_bool('True')
    assert utils.parse_bool('true')

    with pytest.raises(ValueError) as e:
        utils.parse_bool('random_string')


def test_validate_shape():
    with pytest.raises(ValueError):
        utils.validate_shape(np.zeros((3, 2)), 6, 4)

    try:
        utils.validate_shape(np.zeros((3, 2)), 3, 2)
    except ValueError as e:
        assert False, f"Test validate_shape raised an exception {e}"
