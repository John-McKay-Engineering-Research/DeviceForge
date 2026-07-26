from __future__ import annotations

import numpy as np
import pytest

from deviceforge.core import FaceField, Grid


def test_create_face_field() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    field = FaceField(
        name="face_field",
        units="V/m",
        grid=grid,
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
        ],
    )

    assert field.shape == (4,)
    assert field.number_of_faces == 4
    assert field.minimum == pytest.approx(1.0)
    assert field.maximum == pytest.approx(4.0)
    assert field.mean == pytest.approx(2.5)


def test_face_coordinates_are_midpoints() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(2.0,),
        origin=(1.0,),
    )

    field = FaceField(
        name="face_field",
        units="V/m",
        grid=grid,
        values=np.zeros(4),
    )

    np.testing.assert_allclose(
        field.coordinates(),
        [2.0, 4.0, 6.0, 8.0],
    )


def test_face_field_rejects_node_shaped_values() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0,),
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        FaceField(
            name="invalid",
            units="V/m",
            grid=grid,
            values=np.zeros(5),
        )


def test_face_field_values_are_immutable() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0,),
    )

    field = FaceField(
        name="face_field",
        units="V/m",
        grid=grid,
        values=np.zeros(4),
    )

    with pytest.raises(ValueError):
        field.values[0] = 1.0