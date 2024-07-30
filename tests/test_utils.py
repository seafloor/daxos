import pytest
import numpy as np
from daxos.utils import parse_bool, set_numpy_dtype, validate_shape

def test_parse_bool():
    assert parse_bool('True') is True
    assert parse_bool('False') is False
    with pytest.raises(ValueError):
        parse_bool('invalid')

def test_set_numpy_dtype():
    assert set_numpy_dtype('float16') == np.float16
    assert set_numpy_dtype('float32') == np.float32
    assert set_numpy_dtype('float64') == np.float64
    with pytest.raises(ValueError):
        set_numpy_dtype('invalid')

def test_validate_shape():
    data = np.random.rand(10, 5)
    validate_shape(data, 10, 5)  # Should not raise an error
    with pytest.raises(ValueError):
        validate_shape(data, 5, 10)

if __name__ == "__main__":
    pytest.main()
