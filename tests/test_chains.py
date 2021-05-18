import pytest
import numpy as np
from numpy.testing import assert_allclose

from rfidam.chains import FgTransitions


@pytest.fixture
def inventory3():
    return np.array([
        [1, 0, 0, 0],
        [1/3, 2/3, 0, 0],
        [1/9, 2/3, 2/9, 0],
        [1/12, 1/2, 1/4, 1/6],
    ])


def test_foreground_inventory(inventory3):
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': inventory3}

    inv0 = FgTransitions(0, **kwargs).inventory_matrix
    inv1 = FgTransitions(1, **kwargs).inventory_matrix
    inv2 = FgTransitions(2, **kwargs).inventory_matrix
    inv3 = FgTransitions(3, **kwargs).inventory_matrix

    assert inv0.shape == (7, 7)
    assert_allclose(inv0, np.identity(7))

    assert_allclose(inv1.sum(axis=1), np.ones(7))
    assert_allclose(inv1[0], [1/3, 0, 0, 1/2, 0, 0, 1/6])
    assert_allclose(inv1[1:, 1:], np.identity(6))

    assert_allclose(inv2.sum(axis=1), np.ones(7))
    assert_allclose(inv2[1], [1/3, 1/9, 0, 1/6, 1/4, 0, 5/36])
    assert_allclose(inv2[4], [0, 0, 0, 2/3, 1/3, 0, 0])
    for i in range(7):
        if i not in [1, 4]:
            assert_allclose(inv2[i], inv1[i],
                            err_msg=f"line {i} mismatch at inv2")

    assert_allclose(inv3.sum(axis=1), np.ones(7))
    assert_allclose(inv3[2], [1/12, 1/3, 1/12, 1/8, 1/8, 1/8, 1/8])
    assert_allclose(inv3[5], [0, 0, 0, 2/9, 2/3, 1/9, 0])
    for i in range(7):
        if i not in [2, 5]:
            assert_allclose(inv3[i], inv2[i],
                            err_msg=f"line {i} mismatch at inv3")


def test_foreground_target_switch(inventory3):
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': inventory3}

    mat1 = FgTransitions(1, **kwargs).target_switch_matrix
    mat3 = FgTransitions(3, **kwargs).target_switch_matrix

    assert_allclose(mat1, [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    assert_allclose(mat3, [
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])


def test_foreground_power_off():
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': None}

    mat1 = FgTransitions(1, **kwargs).power_off_matrix
    mat3 = FgTransitions(3, **kwargs).power_off_matrix

    assert_allclose(mat1, [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    assert_allclose(mat3, [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])


def test_foreground_arrival_a():
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': None}

    mat1 = FgTransitions(1, **kwargs).arrival_matrix_a
    mat2 = FgTransitions(2, **kwargs).arrival_matrix_a

    assert_allclose(mat1, [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    assert_allclose(mat2, [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])


def test_foreground_arrival_b():
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': None}

    mat1 = FgTransitions(1, **kwargs).arrival_matrix_b
    mat2 = FgTransitions(2, **kwargs).arrival_matrix_b

    assert_allclose(mat1, np.identity(7))
    assert_allclose(mat2, np.identity(7))


def test_foreground_departure():
    kwargs = {'p_id': 0.25, 'n_tags_max': 3, 'inventory_probs': None}

    mat2 = FgTransitions(2, **kwargs).departure_matrix
    mat3 = FgTransitions(3, **kwargs).departure_matrix

    assert_allclose(mat2, [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    assert_allclose(mat3, [
        [1, 0, 0, 0, 0, 0, 0],
        [1/2, 1/2, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1/2, 1/2, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
