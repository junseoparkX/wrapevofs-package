import pandas as pd

from wrapevofs.config import SplitConfig
from wrapevofs.split import parse_split_ratio, train_test_split_frame


def test_parse_split_ratio_presets():
    assert parse_split_ratio("7:3") == 0.3
    assert parse_split_ratio("6:4") == 0.4
    assert parse_split_ratio("8:2") == 0.2


def test_train_test_split_frame_uses_ratio():
    df = pd.DataFrame(
        {
            "x1": range(100),
            "x2": range(100, 200),
            "target": [0, 1] * 50,
        }
    )
    split = train_test_split_frame(df, "target", SplitConfig(ratio="8:2"))
    assert len(split.X_train) == 80
    assert len(split.X_test) == 20
