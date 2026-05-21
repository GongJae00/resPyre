import json
from pathlib import Path

from components.observations.methods import FamilyMultiViewHypothesisSet_Model
from components.observations.semantics import (
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    get_observation_family_semantics,
)
from scripts.materialize_calibrated_multifamily_parh_system import DEFAULT_METHODS


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = list(CANONICAL_OBSERVATION_FAMILY_ORDER)
EXPECTED_DISPLAY = [
    "OF",
    "OF_bridge",
    "DoF",
    "DoF_bridge",
    "P1D_lin",
    "P1D_quad",
    "P1D_cub",
    "P1D_cons",
]


def _base_method_name(entry):
    name = entry if isinstance(entry, str) else entry["name"]
    return str(name).split("__", 1)[0]


def _config_family_blocks(path: str) -> list[list[str]]:
    cfg = json.loads((ROOT / path).read_text())
    names = [_base_method_name(entry) for entry in cfg["methods"][:24]]
    return [names[i : i + len(EXPECTED)] for i in range(0, 24, len(EXPECTED))]


def test_canonical_order_is_the_paper_facing_observation_order() -> None:
    display = [
        str(get_observation_family_semantics(name)["display_name"])
        for name in EXPECTED
    ]
    assert display == EXPECTED_DISPLAY


def test_multichannel_and_materializer_use_the_same_view_order() -> None:
    assert FamilyMultiViewHypothesisSet_Model().component_names == EXPECTED
    assert [family_name for _label, family_name, _group in DEFAULT_METHODS] == EXPECTED


def test_production_configs_group_base_kfstd_and_parh_in_canonical_order() -> None:
    for path in (
        "configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json",
        "configs/mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json",
    ):
        assert _config_family_blocks(path) == [EXPECTED, EXPECTED, EXPECTED]
