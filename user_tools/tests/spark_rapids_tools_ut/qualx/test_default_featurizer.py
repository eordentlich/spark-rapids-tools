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

"""Tests for the default QualX featurizer."""

import logging

import pandas as pd

from spark_rapids_tools.tools.qualx.featurizers import default
from spark_rapids_tools.tools.qualx.stage_type import (
    STAGE_TYPE_COL,
    STAGE_TYPE_INPUT_SCAN,
    STAGE_TYPE_NO_INPUT_SCAN,
)


def _sql_to_stage() -> pd.DataFrame:
    return pd.DataFrame({
        'sqlID': [1, 1],
        'stageId': [10, 11],
        'SQL Nodes(IDs)': ['GpuHashAggregate(1)', 'GpuProject(2)'],
    })


def test_build_stage_type_map_uses_stage_scan_time_metric(monkeypatch):
    """Scan-time stage metrics should classify stages even when plan nodes omit scans."""
    monkeypatch.setattr(default, 'is_duration_sum_stage_type_enabled', lambda: True)

    sql_ops_metrics = pd.DataFrame({
        'appId': ['app-1'],
        'sqlID': [1],
        'nodeName': ['GpuHashAggregate'],
        'stageIds': ['11'],
    })
    stage_metrics = pd.DataFrame({
        'stageId': [10],
        'name': ['scan time'],
        'total': [123],
    })

    result = default._build_stage_type_map(sql_ops_metrics, _sql_to_stage(), 'app-1', stage_metrics)

    assert result.loc[result['stageId'].eq(10), STAGE_TYPE_COL].iloc[0] == STAGE_TYPE_INPUT_SCAN
    assert result.loc[result['stageId'].eq(11), STAGE_TYPE_COL].iloc[0] == STAGE_TYPE_NO_INPUT_SCAN


def test_build_stage_type_map_handles_no_scan_signals(monkeypatch):
    """Missing scan signals should produce non-scan rows instead of raising KeyError."""
    monkeypatch.setattr(default, 'is_duration_sum_stage_type_enabled', lambda: True)

    sql_ops_metrics = pd.DataFrame({
        'appId': ['app-1'],
        'sqlID': [1],
        'nodeName': ['GpuHashAggregate'],
        'stageIds': ['11'],
    })
    stage_metrics = pd.DataFrame(columns=['stageId', 'name', 'total'])

    result = default._build_stage_type_map(sql_ops_metrics, _sql_to_stage(), 'app-1', stage_metrics)

    assert set(result['stageId']) == {10, 11}
    assert set(result[STAGE_TYPE_COL]) == {STAGE_TYPE_NO_INPUT_SCAN}


def test_stage_type_rollup_check_allows_small_drift(caplog):
    """StageType rollup checks should allow differences within the configured tolerance."""
    caplog.set_level(logging.WARNING, logger=default.logger.name)
    source_tbl = pd.DataFrame({
        'appId': ['app-1'],
        'appName': ['dataset'],
        'sqlID': [1],
        'duration_sum': [100.0],
        'numTasks_sum': [10.0],
        'duration_max': [70.0],
    })
    stage_type_tbl = pd.DataFrame({
        'appId': ['app-1', 'app-1'],
        'appName': ['dataset', 'dataset'],
        'sqlID': [1, 1],
        STAGE_TYPE_COL: [STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN],
        'duration_sum': [40.0, 60.05],
        'numTasks_sum': [4.0, 6.0],
        'duration_max': [40.0, 60.0],
    })

    default._warn_stage_type_rollup_mismatches(
        source_tbl,
        stage_type_tbl,
        {'duration_sum': 'sum', 'numTasks_sum': 'sum', 'duration_max': 'max'},
    )

    assert 'StageType rollup mismatch' not in caplog.text


def test_stage_type_rollup_check_warns_on_significant_mismatch(caplog):
    """StageType rollup checks should warn when additive metrics exceed tolerance."""
    caplog.set_level(logging.WARNING, logger=default.logger.name)
    source_tbl = pd.DataFrame({
        'appId': ['app-1'],
        'appName': ['dataset'],
        'sqlID': [1],
        'duration_sum': [100.0],
        'numTasks_sum': [10.0],
        'duration_max': [70.0],
    })
    stage_type_tbl = pd.DataFrame({
        'appId': ['app-1', 'app-1'],
        'appName': ['dataset', 'dataset'],
        'sqlID': [1, 1],
        STAGE_TYPE_COL: [STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN],
        'duration_sum': [40.0, 62.0],
        'numTasks_sum': [4.0, 6.0],
        'duration_max': [40.0, 80.0],
    })

    default._warn_stage_type_rollup_mismatches(
        source_tbl,
        stage_type_tbl,
        {'duration_sum': 'sum', 'numTasks_sum': 'sum', 'duration_max': 'max'},
    )

    assert 'StageType rollup mismatch' in caplog.text
    assert 'duration_sum' in caplog.text
    assert 'duration_max' not in caplog.text
