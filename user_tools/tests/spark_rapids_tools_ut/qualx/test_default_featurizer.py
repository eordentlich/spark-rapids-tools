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

"""Test default QualX featurizer."""

import pandas as pd

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.featurizers import default as default_featurizer
from ..conftest import SparkRapidsToolsUT


BYTE_SUM_COLUMNS = [
    'diskBytesSpilled_sum',
    'memoryBytesSpilled_sum',
    'input_bytesRead_sum',
    'output_bytesWritten_sum',
    'sr_localBytesRead_sum',
    'sr_remoteBytesRead_sum',
    'sr_remoteBytesReadToDisk_sum',
    'sr_totalBytesRead_sum',
    'sw_bytesWritten_sum',
]

TIME_SUM_COLUMNS = [
    'executorCPUTime_sum',
    'executorDeserializeCPUTime_sum',
    'executorDeserializeTime_sum',
    'executorRunTime_sum',
    'jvmGCTime_sum',
    'sr_fetchWaitTime_sum',
    'sw_writeTime_sum',
]

OTHER_SUM_COLUMNS = [
    'input_recordsRead_sum',
    'output_recordsWritten_sum',
    'resultSerializationTime_sum',
    'sw_recordsWritten_sum',
]


def _stage_metrics_row(sql_id: int, supported: bool, duration_sum: int, num_tasks: int) -> dict:
    row = {
        'appId': 'app-1',
        'appName': 'dataset',
        'sqlID': sql_id,
        'Exec Is Supported': supported,
        'duration_sum': duration_sum,
        'duration_min': duration_sum,
        'duration_max': duration_sum,
        'numTasks_sum': num_tasks,
        'peakExecutionMemory_max': 0,
        'resultSize_max': 0,
    }
    row.update({col: 0 for col in BYTE_SUM_COLUMNS})
    row.update({col: 0 for col in TIME_SUM_COLUMNS})
    row.update({col: 0 for col in OTHER_SUM_COLUMNS})
    row['input_bytesRead_sum'] = 1
    row['sr_fetchWaitTime_sum'] = 1
    row['sw_writeTime_sum'] = 1
    return row


def _app_row(sql_id: int, duration: int) -> dict:
    return {
        'appId': 'app-1',
        'appDuration': 350,
        'sqlID': sql_id,
        'Duration': duration,
        'description': f'sql-{sql_id}',
        'sparkRuntime': 'SPARK',
        'sparkVersion': '3.5.0',
        'pluginEnabled': False,
        'resourceProfileId': 0,
        'numExecutors': 1,
        'executorCores': 4,
        'maxMem': 1,
        'maxOnHeapMem': 1,
        'maxOffHeapMem': 0,
        'executorMemory': 1,
        'numGpusPerExecutor': 0,
        'executorOffHeap': 0,
        'taskCpu': 1.0,
        'taskGpu': 0.0,
        'startTime': sql_id,
    }


def _tables() -> dict:
    return {
        'app_tbl': pd.DataFrame([
            _app_row(1, 150),
            _app_row(2, 200),
        ]),
        'ops_tbl': pd.DataFrame([{
            'appId': 'app-1',
            'appName': 'dataset',
            'sqlID': 1,
            'nodeID': 10,
            'nodeName': 'Project',
            'metricType': 'timing',
            'max': 0,
            'name': 'duration',
            'total': 0,
        }]),
        'job_stage_agg_tbl': pd.DataFrame([
            _stage_metrics_row(1, True, 100, 10),
            _stage_metrics_row(1, False, 50, 5),
            _stage_metrics_row(2, False, 200, 20),
        ]),
        'wholestage_tbl': pd.DataFrame(columns=['appId', 'sqlID', 'nodeID', 'Child Node']),
        'failed_tasks_tbl': pd.DataFrame(columns=['appName', 'appId', 'sqlID', 'failed_tasks']),
        'ds_tbl': pd.DataFrame([
            {'appId': 'app-1', 'sqlID': 1, 'data_size': 100, 'scan_time': 10, 'decode_time': 5},
            {'appId': 'app-1', 'sqlID': 2, 'data_size': 200, 'scan_time': 20, 'decode_time': 10},
        ]),
    }


def _extract_features(monkeypatch) -> pd.DataFrame:
    toc = pd.DataFrame({'appId': ['app-1'], 'ds_name': ['dataset']})
    node_level_supp = pd.DataFrame({'present': [True]})
    monkeypatch.setattr(default_featurizer, 'load_csv_files', lambda *args, **kwargs: _tables())
    return default_featurizer.extract_raw_features(
        toc,
        node_level_supp=node_level_supp,
        qualtool_filter='stage',
        remove_failed_sql=False,
    )


class TestDefaultFeaturizer(SparkRapidsToolsUT):
    """Test class for default QualX featurizer."""

    def test_stage_filter_aggregates_features_over_supported_stages(self, monkeypatch):
        """Unsupported-only SQLs should not produce feature rows for XGBoost."""
        features = _extract_features(monkeypatch)

        assert features['sqlID'].tolist() == [1]

        row = features.iloc[0]
        assert row['duration_sum'] == 100
        assert row['numTasks_sum'] == 10
        assert row['fraction_supported'] == 100 / 150

    def test_duration_sum_label_preserves_total_duration(self, monkeypatch):
        """duration_sum remains the total label while other stage metrics use supported stages."""
        with monkeypatch.context() as mp:
            mp.setenv('QUALX_LABEL', 'duration_sum')
            get_config(reload=True)
            features = _extract_features(mp)

        get_config(reload=True)

        assert features['sqlID'].tolist() == [1]

        row = features.iloc[0]
        assert row['duration_sum'] == 150
        assert row['duration_ratio'] == 100 / 150
        assert row['numTasks_sum'] == 10
        assert row['fraction_supported'] == 100 / 150
