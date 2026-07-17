import numpy as np
import pandas as pd

from src.deployment import build_features


def create_raw_frame() -> pd.DataFrame:
    data = {
        "Time": [
            0.0,
            21600.0,
            43200.0,
        ],
        "Amount": [
            0.0,
            9.0,
            99.0,
        ],
    }

    for index in range(1, 29):
        data[f"V{index}"] = [
            0.0,
            float(index),
            float(-index),
        ]

    return pd.DataFrame(data)


def test_build_features_preserves_order() -> None:
    raw = create_raw_frame()

    names = [
        *[
            f"V{index}"
            for index in range(1, 29)
        ],
        "Amount_log1p",
        "Time_sin",
        "Time_cos",
    ]

    features = build_features(
        raw,
        names,
    )

    assert list(features.columns) == names
    assert features.shape == (3, 31)
    assert not features.isna().any().any()


def test_amount_log_transform() -> None:
    raw = create_raw_frame()

    features = build_features(
        raw,
        ["Amount_log1p"],
    )

    expected = np.log1p(
        raw["Amount"].to_numpy()
    )

    np.testing.assert_allclose(
        features[
            "Amount_log1p"
        ].to_numpy(),
        expected,
    )


def test_cyclic_time_features() -> None:
    raw = create_raw_frame()

    features = build_features(
        raw,
        [
            "Time_sin",
            "Time_cos",
        ],
    )

    np.testing.assert_allclose(
        features.loc[0, "Time_sin"],
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        features.loc[0, "Time_cos"],
        1.0,
        atol=1e-12,
    )