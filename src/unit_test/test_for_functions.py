from unit_test.combination_function import combination
import itertools
import pytest

def test_empty_set():
    assert combination([]) == [[]]


def test_unique_element():
    assert combination(["a"]) == [[], ["a"]]


def test_two_elements():

    assert combination(["a", "b"]) == [[], ["b"], ["a"], ["b", "a"]]
