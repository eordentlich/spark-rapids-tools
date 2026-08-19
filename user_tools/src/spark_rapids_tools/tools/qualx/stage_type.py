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

"""Stage-type feature constants and helpers for duration_sum QualX models."""

import pandas as pd

STAGE_TYPE_COL = 'stageType'
STAGE_TYPE_INPUT_SCAN = 0
STAGE_TYPE_NO_INPUT_SCAN = 1
STAGE_TYPE_VALUES = (STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN)

# Input-scan stages are identified by SQL plan nodes such as "Scan parquet" and "GpuScan parquet".
# The negative look-behind avoids treating names such as "LocalTableScan" as input scans.
SCAN_NODE_PATTERN = r'(?<![A-Za-z])(?:Gpu)?Scan\b'


def normalize_stage_type_column(
    df: pd.DataFrame,
    *,
    context: str = 'DataFrame',
) -> pd.DataFrame:
    """Return a dataframe whose stageType column is a numeric int64 binary feature."""
    if df.empty or STAGE_TYPE_COL not in df.columns:
        return df

    normalized = pd.to_numeric(df[STAGE_TYPE_COL], errors='coerce')
    invalid_mask = normalized.isna()
    if invalid_mask.any():
        invalid_values = df.loc[invalid_mask, STAGE_TYPE_COL].drop_duplicates().head(5).tolist()
        raise ValueError(
            f'{context} contains non-numeric {STAGE_TYPE_COL} values: {invalid_values}'
        )

    non_integer_mask = normalized.mod(1).ne(0)
    if non_integer_mask.any():
        invalid_values = df.loc[non_integer_mask, STAGE_TYPE_COL].drop_duplicates().head(5).tolist()
        raise ValueError(
            f'{context} contains non-integer {STAGE_TYPE_COL} values: {invalid_values}'
        )

    normalized = normalized.astype('int64')
    unexpected_values = sorted(set(normalized.unique()) - set(STAGE_TYPE_VALUES))
    if unexpected_values:
        raise ValueError(
            f'{context} contains unexpected {STAGE_TYPE_COL} values: {unexpected_values}; '
            f'expected {list(STAGE_TYPE_VALUES)}'
        )

    result = df.copy()
    result[STAGE_TYPE_COL] = normalized
    return result


def validate_stage_type_splits(
    df: pd.DataFrame,
    *,
    context: str = 'DataFrame',
) -> None:
    """Raise if stageType rows for the same SQL record have different splits."""
    required_cols = {STAGE_TYPE_COL, 'appId', 'sqlID', 'split'}
    if df.empty or not required_cols.issubset(df.columns):
        return

    group_cols = ['appId', 'sqlID']
    split_counts = df.groupby(group_cols, dropna=False)['split'].nunique(dropna=False)
    inconsistent = split_counts[split_counts > 1]
    if inconsistent.empty:
        return

    examples = inconsistent.reset_index()[group_cols].head(5).to_dict('records')
    raise ValueError(
        f'{context} contains stageType rows with inconsistent split assignments for '
        f'{len(inconsistent)} SQL records; examples: {examples}'
    )
