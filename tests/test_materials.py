import pytest

from deviceforge.physics import (
    GENERIC_CONDUCTOR,
    SILICON,
    SILICON_DIOXIDE,
    Material,
)


def test_create_semiconductor_material() -> None:
    material = Material(
        name="Test semiconductor",
        material_type="semiconductor",
        relative_permittivity=12.0,
        band_gap_ev=1.1,
        electron_affinity_ev=4.0,
        electron_mobility=0.1,
        hole_mobility=0.05,
        thermal_conductivity=100.0,
    )

    assert material.name == "Test semiconductor"
    assert material.material_type == "semiconductor"
    assert material.relative_permittivity == pytest.approx(12.0)
    assert material.is_semiconductor
    assert not material.is_dielectric
    assert not material.is_conductor


def test_material_type_is_normalised() -> None:
    material = Material(
        name="Oxide",
        material_type="  DIELECTRIC  ",
        relative_permittivity=3.9,
    )

    assert material.material_type == "dielectric"
    assert material.is_dielectric


def test_optional_properties_can_be_omitted() -> None:
    material = Material(
        name="Generic material",
        material_type="conductor",
        relative_permittivity=1.0,
    )

    assert material.band_gap_ev is None
    assert material.electron_mobility is None


def test_predefined_silicon() -> None:
    assert SILICON.name == "Silicon"
    assert SILICON.is_semiconductor
    assert SILICON.relative_permittivity == pytest.approx(11.7)
    assert SILICON.band_gap_ev == pytest.approx(1.12)


def test_predefined_silicon_dioxide() -> None:
    assert SILICON_DIOXIDE.is_dielectric
    assert SILICON_DIOXIDE.relative_permittivity == pytest.approx(3.9)


def test_predefined_generic_conductor() -> None:
    assert GENERIC_CONDUCTOR.is_conductor


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_name_raises_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        Material(
            name=name,
            material_type="semiconductor",
            relative_permittivity=11.7,
        )


@pytest.mark.parametrize(
    "material_type",
    [
        "",
        "metallic",
        "unknown",
        "semiconducting",
    ],
)
def test_invalid_material_type_raises_value_error(
    material_type: str,
) -> None:
    with pytest.raises(ValueError, match="Material type"):
        Material(
            name="Invalid material",
            material_type=material_type,
            relative_permittivity=1.0,
        )


@pytest.mark.parametrize(
    "relative_permittivity",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_permittivity_raises_value_error(
    relative_permittivity: float,
) -> None:
    with pytest.raises(ValueError, match="permittivity"):
        Material(
            name="Invalid material",
            material_type="dielectric",
            relative_permittivity=relative_permittivity,
        )


@pytest.mark.parametrize(
    ("property_name", "invalid_value"),
    [
        ("band_gap_ev", -1.0),
        ("electron_affinity_ev", -1.0),
        ("electron_mobility", -0.1),
        ("hole_mobility", -0.1),
        ("thermal_conductivity", -1.0),
        ("band_gap_ev", float("nan")),
        ("electron_mobility", float("inf")),
    ],
)
def test_invalid_optional_property_raises_value_error(
    property_name: str,
    invalid_value: float,
) -> None:
    parameters = {
        "name": "Invalid material",
        "material_type": "semiconductor",
        "relative_permittivity": 11.7,
        property_name: invalid_value,
    }

    with pytest.raises(ValueError, match=property_name):
        Material(**parameters)