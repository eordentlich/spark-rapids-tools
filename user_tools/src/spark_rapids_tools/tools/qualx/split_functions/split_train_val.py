# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Default split functions."""

import hashlib
import random
import pandas as pd


def _stable_group_split(group_key, seed: int, val_pct: float) -> str:
    """Assign a deterministic split that is independent of dataframe shape and order."""
    key = '\x1f'.join(str(value) for value in group_key)
    digest = hashlib.sha256(f'{seed}\x1f{key}'.encode('utf-8')).digest()
    score = int.from_bytes(digest[:8], byteorder='big') / 2**64
    return 'val' if score < val_pct else 'train'


def split_function(features: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Randomly split all rows into 'train' and 'val' sets.

    Parameters
    ----------
    features: pd.DataFrame
        Input dataframe
    seed: int
        Seed for random number generator
    val_pct: float
        Percentage of all rows to use for validation

    Returns
    -------
    pd.DataFrame
    """
    seed = kwargs.get('seed', 0)
    val_pct = kwargs.get('val_pct', 0.2)

    if 'stageType' in features.columns and {'appId', 'sqlID'}.issubset(features.columns):
        # Both stageType records describe the same SQL sample. Keep them in the same split,
        # and make the assignment stable when raw and qualification-filtered data differ.
        for group_key, group in features.groupby(['appId', 'sqlID'], dropna=False, sort=False):
            missing_indices = group.index[group['split'].isna()] if 'split' in group else group.index
            if missing_indices.empty:
                continue

            existing_splits = group['split'].dropna().unique() if 'split' in group else []
            if len(existing_splits) > 1:
                raise ValueError(
                    f'Conflicting preassigned splits for stageType SQL record {group_key}: '
                    f'{sorted(existing_splits)}'
                )
            split = (
                existing_splits[0]
                if len(existing_splits) == 1
                else _stable_group_split(group_key, seed, val_pct)
            )
            features.loc[missing_indices, 'split'] = split
        return features

    if 'split' in features.columns:
        # if 'split' already present, just modify the NaN rows
        # otherwise, modify all rows
        indices = features.index[features['split'].isna()].tolist()
    else:
        indices = features.index.tolist()

    num_rows = len(indices)
    random.Random(seed).shuffle(indices)

    # split remaining rows into train/val sets
    features.loc[indices[0: int(val_pct * num_rows)], 'split'] = 'val'
    features.loc[indices[int(val_pct * num_rows):], 'split'] = 'train'
    return features
