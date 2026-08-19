# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for stageType helpers."""

import pandas as pd
import pytest

from spark_rapids_tools.tools.qualx.stage_type import (
    STAGE_TYPE_COL,
    STAGE_TYPE_INPUT_SCAN,
    STAGE_TYPE_NO_INPUT_SCAN,
    normalize_stage_type_column,
)


def test_normalize_stage_type_column_converts_object_binary_values():
    """Object-typed binary values should become a numeric model feature."""
    df = pd.DataFrame({
        STAGE_TYPE_COL: pd.Series(
            [
                str(STAGE_TYPE_INPUT_SCAN),
                str(STAGE_TYPE_NO_INPUT_SCAN),
                STAGE_TYPE_INPUT_SCAN,
                STAGE_TYPE_NO_INPUT_SCAN,
            ],
            dtype=object,
        )
    })

    result = normalize_stage_type_column(df)

    assert pd.api.types.is_integer_dtype(result[STAGE_TYPE_COL])
    assert result[STAGE_TYPE_COL].tolist() == [
        STAGE_TYPE_INPUT_SCAN,
        STAGE_TYPE_NO_INPUT_SCAN,
        STAGE_TYPE_INPUT_SCAN,
        STAGE_TYPE_NO_INPUT_SCAN,
    ]


def test_normalize_stage_type_column_rejects_invalid_values():
    """Invalid stageType values should fail before XGBoost sees the dataframe."""
    df = pd.DataFrame({STAGE_TYPE_COL: ['0', 'input-scan']})

    with pytest.raises(ValueError, match=f'non-numeric {STAGE_TYPE_COL}'):
        normalize_stage_type_column(df)
