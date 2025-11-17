"""Tests for the sequence validator (Sequence[T] types)."""

from collections.abc import Sequence

import pytest

from pydantic_core import SchemaValidator, ValidationError, core_schema

from ..conftest import PyAndJson


@pytest.mark.parametrize(
    'input_value,expected',
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        ([1, 2, '3'], [1, 2, 3]),
        ((1, 2, '3'), (1, 2, 3)),
        ([], []),
        ((), ()),
    ],
    ids=repr,
)
def test_sequence_basic(py_and_json: PyAndJson, input_value, expected):
    """Test basic sequence validation with lists and tuples."""
    # Use json_or_python_schema to handle both JSON and Python inputs
    v = py_and_json(
        core_schema.json_or_python_schema(
            json_schema=core_schema.list_schema(items_schema=core_schema.int_schema()),
            python_schema=core_schema.chain_schema(
                [
                    core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                    core_schema.sequence_schema(items_schema=core_schema.int_schema()),
                ]
            ),
        )
    )
    result = v.validate_test(input_value)
    # For JSON, tuples become lists, so we only check type preservation for Python
    if v.validator_type == 'python':
        assert result == expected
        assert type(result) == type(expected)
    else:
        # For JSON, we just check the values match (tuples become lists)
        # Convert expected tuple to list for comparison, otherwise use as-is
        expected_for_json = list(expected) if isinstance(expected, tuple) else expected
        assert result == expected_for_json


def test_sequence_list_preserved():
    """Test that lists are preserved as lists."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    result = v.validate_python([1, 2, 3])
    assert result == [1, 2, 3]
    assert isinstance(result, list)
    assert not isinstance(result, tuple)


def test_sequence_tuple_preserved():
    """Test that tuples are preserved as tuples."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    result = v.validate_python((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)
    assert not isinstance(result, list)


@pytest.mark.parametrize(
    'input_value,expected_error',
    [
        # Strings and bytes ARE instances of Sequence, but our validator rejects them
        ('abc', 'instances are not allowed as a Sequence value'),
        (b'abc', 'instances are not allowed as a Sequence value'),
        # These are NOT instances of Sequence
        (123, 'Input should be an instance of Sequence'),
        ({'a': 1}, 'Input should be an instance of Sequence'),
    ],
    ids=repr,
)
def test_sequence_isinstance_check(input_value, expected_error):
    """Test that isinstance check happens before sequence validation."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    with pytest.raises(ValidationError) as exc_info:
        v.validate_python(input_value)
    assert expected_error in str(exc_info.value)


def test_sequence_rejects_string():
    """Test that strings are rejected even though they're sequences."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    with pytest.raises(ValidationError) as exc_info:
        v.validate_python('abc')
    errors = exc_info.value.errors(include_url=False)
    assert len(errors) > 0
    # The error should mention that string instances are not allowed
    error_msg = str(errors[0])
    assert 'str' in error_msg.lower() or 'string' in error_msg.lower()


def test_sequence_rejects_bytes():
    """Test that bytes are rejected even though they're sequences."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    with pytest.raises(ValidationError) as exc_info:
        v.validate_python(b'abc')
    errors = exc_info.value.errors(include_url=False)
    assert len(errors) > 0
    # The error should mention that bytes instances are not allowed
    error_msg = str(errors[0])
    assert 'bytes' in error_msg.lower()


@pytest.mark.parametrize(
    'input_value,expected',
    [
        ([1, 2, 3], [1, 2, 3]),
        (['1', '2', '3'], [1, 2, 3]),
        (['1', 2, '3'], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        (('1', '2', '3'), (1, 2, 3)),
    ],
    ids=repr,
)
def test_sequence_item_validation(input_value, expected):
    """Test that items in the sequence are validated."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    result = v.validate_python(input_value)
    assert result == expected
    assert type(result) == type(expected)


def test_sequence_item_validation_error():
    """Test that item validation errors are properly reported."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    with pytest.raises(ValidationError) as exc_info:
        v.validate_python([1, 2, 'not an int'])
    errors = exc_info.value.errors(include_url=False)
    assert len(errors) > 0
    # Should have an error for the invalid item at index 2
    assert any('not an int' in str(err) or err.get('loc') == (2,) for err in errors)


def test_sequence_no_items_schema():
    """Test sequence validation without item schema (any items allowed)."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(),
            ]
        )
    )
    result = v.validate_python([1, '2', 3.0, [4, 5]])
    assert result == [1, '2', 3.0, [4, 5]]
    assert isinstance(result, list)

    result = v.validate_python((1, '2', 3.0))
    assert result == (1, '2', 3.0)
    assert isinstance(result, tuple)


def test_sequence_nested():
    """Test nested sequences."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(
                    items_schema=core_schema.chain_schema(
                        [
                            core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                            core_schema.sequence_schema(items_schema=core_schema.int_schema()),
                        ]
                    )
                ),
            ]
        )
    )
    result = v.validate_python([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result)

    result = v.validate_python(((1, 2), (3, 4)))
    assert result == ((1, 2), (3, 4))
    assert isinstance(result, tuple)
    assert all(isinstance(item, tuple) for item in result)


def test_sequence_empty():
    """Test empty sequences."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    assert v.validate_python([]) == []
    assert v.validate_python(()) == ()


def test_sequence_with_str_items():
    """Test sequence with string items."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.str_schema()),
            ]
        )
    )
    result = v.validate_python(['a', 'b', 'c'])
    assert result == ['a', 'b', 'c']
    assert isinstance(result, list)

    result = v.validate_python(('a', 'b', 'c'))
    assert result == ('a', 'b', 'c')
    assert isinstance(result, tuple)


def test_sequence_with_float_items():
    """Test sequence with float items."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.float_schema()),
            ]
        )
    )
    result = v.validate_python([1, 2, 3])
    assert result == [1.0, 2.0, 3.0]
    assert isinstance(result, list)

    result = v.validate_python((1, 2, 3))
    assert result == (1.0, 2.0, 3.0)
    assert isinstance(result, tuple)


def test_sequence_range_object():
    """Test that range objects (which are sequences) can be validated."""
    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    # Range is a sequence, so it should pass isinstance check
    # But we convert it to a list for validation, so it becomes a list
    result = v.validate_python(range(3))
    # Range gets converted to list during validation
    assert isinstance(result, list)
    assert len(result) == 3


def test_sequence_custom_sequence_type():
    """Test with a custom sequence-like type."""

    # Create a simple custom sequence
    class MySequence:
        def __init__(self, items):
            self.items = list(items)

        def __iter__(self):
            return iter(self.items)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    # Register it as a Sequence subclass
    import collections.abc

    collections.abc.Sequence.register(MySequence)

    v = SchemaValidator(
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                core_schema.sequence_schema(items_schema=core_schema.int_schema()),
            ]
        )
    )
    my_seq = MySequence([1, 2, 3])
    result = v.validate_python(my_seq)
    # Custom sequences get converted to list
    assert isinstance(result, list)
    assert result == [1, 2, 3]


def test_sequence_json():
    """Test sequence validation with JSON input."""
    v = SchemaValidator(
        core_schema.json_or_python_schema(
            json_schema=core_schema.list_schema(items_schema=core_schema.int_schema()),
            python_schema=core_schema.chain_schema(
                [
                    core_schema.is_instance_schema(Sequence, cls_repr='Sequence'),
                    core_schema.sequence_schema(items_schema=core_schema.int_schema()),
                ]
            ),
        )
    )
    result = v.validate_json('[1, "2", 3]')
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_sequence_name():
    """Test that the validator has the correct name."""
    v = SchemaValidator(core_schema.sequence_schema(items_schema=core_schema.int_schema()))
    # The name should include the item type
    title = v.title if isinstance(v.title, str) else str(v.title)
    assert 'sequence' in title.lower() or 'int' in title.lower()
