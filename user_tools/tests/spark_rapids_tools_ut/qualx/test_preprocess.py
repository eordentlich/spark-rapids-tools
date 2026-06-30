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

"""Test qualx_preprocess module"""
# pylint: disable=protected-access
import os
import json
import tempfile
from unittest.mock import patch

import pandas as pd

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.featurizers import default as default_featurizer
from spark_rapids_tools.tools.qualx.preprocess import (
    expected_raw_features,
    impute,
    get_alignment,
    get_featurizers,
    get_modifiers,
    infer_app_meta,
    load_qtool_execs,
    load_datasets
)
from ..conftest import SparkRapidsToolsUT


class TestPreprocess(SparkRapidsToolsUT):
    """Test class for qualx_preprocess module"""
    def test_impute(self):
        # Test impute function
        input_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        expected_features = expected_raw_features()
        imputed_df = impute(input_df, expected_features)
        df_columns = set(imputed_df.columns)

        # should not have extra columns
        assert 'col1' not in df_columns
        assert 'col2' not in df_columns

        # should have all expected raw features
        assert df_columns == expected_features

        # fraction_supported should be 1.0
        assert imputed_df['fraction_supported'].iloc[0] == 1.0

        # all other columns should be 0.0
        assert imputed_df[list(df_columns - {'fraction_supported'})].iloc[0].sum() == 0.0

    def test_gpu_max_task_metrics_pivot(self):
        # Test gpu_sql_level_aggregated_task_metrics pivot from rows to columns
        input_df = pd.DataFrame({
            'sqlId': [1, 1, 1, 2],
            'metricName': [
                'gpuMaxDeviceMemoryBytes',
                'gpuMaxTaskFootprint',
                'gpuTime',
                'gpuMaxConcurrentGpuTasks',
            ],
            'unit': ['bytes', 'count', 'ms', 'count'],
            'sum': ['', '', 50, ''],
            'max': [10, 512, 25, 3],
            'avg': ['', '', 5, ''],
        })

        result = default_featurizer._load_gpu_max_task_metrics(input_df, 'app-1')

        assert set(default_featurizer.GPU_MAX_TASK_METRIC_FEATURES).issubset(result.columns)
        assert 'gpuTime' not in result.columns
        sql_1 = result.loc[result['sqlID'] == 1].iloc[0]
        assert sql_1['appId'] == 'app-1'
        assert sql_1['gpuMaxDeviceMemoryBytes'] == 10
        assert sql_1['gpuMaxTaskFootprint'] == 512
        sql_2 = result.loc[result['sqlID'] == 2].iloc[0]
        assert sql_2['gpuMaxConcurrentGpuTasks'] == 3

    def test_gpu_max_task_metrics_expected_features_are_gated(self):
        # Test expected feature columns are included only when the config gate is enabled
        with patch.object(default_featurizer, 'is_gpu_max_task_metrics_enabled', return_value=False):
            disabled_features = default_featurizer.get_expected_raw_features()
        with patch.object(default_featurizer, 'is_gpu_max_task_metrics_enabled', return_value=True):
            enabled_features = default_featurizer.get_expected_raw_features()

        assert 'gpuMaxDeviceMemoryBytes' not in disabled_features
        assert default_featurizer.GPU_MAX_TASK_METRIC_FEATURES.issubset(enabled_features)

    def test_gpu_max_task_metrics_fill_missing_app_with_negative_one(self):
        # Test missing gpu_sql_level_aggregated_task_metrics values are injected as -1
        gpu_max_features = sorted(default_featurizer.GPU_MAX_TASK_METRIC_FEATURES)
        full_tbl = pd.DataFrame({
            'appId': ['app-1', 'app-2'],
            'sqlID': [1, 7],
            'Duration': [100, 200],
        })
        metrics_row = {feature: None for feature in gpu_max_features}
        metrics_row.update({
            'appId': 'app-1',
            'sqlID': 1,
            'gpuMaxDeviceMemoryBytes': 10,
        })
        metrics_tbl = pd.DataFrame([metrics_row], columns=['appId', 'sqlID'] + gpu_max_features)

        result = default_featurizer._add_gpu_max_task_metrics(full_tbl, metrics_tbl)

        app_1 = result.loc[result['appId'] == 'app-1'].iloc[0]
        app_2 = result.loc[result['appId'] == 'app-2'].iloc[0]
        assert app_1['gpuMaxDeviceMemoryBytes'] == 10
        assert app_1['gpuMaxTaskFootprint'] == -1
        assert app_2[gpu_max_features].eq(-1).all()

    def test_get_alignment(self):
        # Test get_alignment function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            # Create temporary alignment CSV file
            test_alignment = pd.DataFrame(
                [
                    ['app1', 'sql1', 'app3', 'sql3'],
                    ['app2', 'sql2', 'app4', 'sql4']
                ],
                columns=['appId_cpu', 'sqlID_cpu', 'appId_gpu', 'sqlID_gpu'],
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                alignment_file = os.path.join(temp_dir, 'alignment.csv')
                test_alignment.to_csv(alignment_file, index=False)

                mock_config.return_value.alignment_dir = temp_dir

                # Call function
                align_df = get_alignment()

                # Verify results
                assert isinstance(align_df, pd.DataFrame)
                assert len(align_df) == 2
                assert list(align_df.columns) == ['appId_cpu', 'sqlID_cpu', 'appId_gpu', 'sqlID_gpu']
                assert list(align_df['appId_cpu']) == ['app1', 'app2']
                assert list(align_df['sqlID_cpu']) == ['sql1', 'sql2']
                assert list(align_df['appId_gpu']) == ['app3', 'app4']
                assert list(align_df['sqlID_gpu']) == ['sql3', 'sql4']

    def test_get_alignment_app_id_only(self):
        # Test get_alignment function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            # Create temporary alignment CSV file
            test_alignment = pd.DataFrame(
                [
                    ['app1', 'app3'],
                    ['app2', 'app4']
                ],
                columns=['appId_cpu', 'appId_gpu'],
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                alignment_file = os.path.join(temp_dir, 'alignment.csv')
                test_alignment.to_csv(alignment_file, index=False)

                mock_config.return_value.alignment_dir = temp_dir

                # Call function
                align_df = get_alignment()

                # Verify results
                assert isinstance(align_df, pd.DataFrame)
                assert len(align_df) == 2
                assert list(align_df.columns) == ['appId_cpu', 'appId_gpu']
                assert list(align_df['appId_cpu']) == ['app1', 'app2']
                assert list(align_df['appId_gpu']) == ['app3', 'app4']

    def test_get_featurizers(self):
        # Test get_featurizers function
        get_config(reload=True)
        result = get_featurizers()

        # Verify results
        assert isinstance(result, list)
        assert len(result) >= 2
        assert all(callable(f.extract_raw_features) for f in result)

        # Test caching
        with patch('spark_rapids_tools.tools.qualx.util.load_plugin') as mock_load_plugin:
            result2 = get_featurizers()
            mock_load_plugin.assert_not_called()
            assert result == result2

    def test_get_modifiers(self):
        # Test get_modifiers function
        with patch('spark_rapids_tools.tools.qualx.preprocess.get_config') as mock_config:
            mock_config.return_value.modifiers = ['align_sql_id.py']
            result = get_modifiers()

            # Verify results
            assert isinstance(result, list)
            assert len(result) >= 1
            assert all(callable(f.modify) for f in result)

            # Test caching
            with patch('spark_rapids_tools.tools.qualx.util.load_plugin') as mock_load_plugin:
                result2 = get_modifiers()
                mock_load_plugin.assert_not_called()
                assert result == result2

    def test_infer_app_meta(self):
        # Test infer_app_meta function
        with patch('spark_rapids_tools.tools.qualx.preprocess.find_eventlogs') as mock_find_eventlogs:
            # Setup mock eventlogs
            mock_find_eventlogs.return_value = [
                '/path/to/job1/eventlogs/CPU/app1',
                '/path/to/job1/eventlogs/GPU/app2',
                '/path/to/job2/eventlogs/CPU/app3',
                '/path/to/job2/eventlogs/GPU/app4',
            ]

            # Call function
            result = infer_app_meta(['/path/to/job1', '/path/to/job2'])

            # Verify results
            assert isinstance(result, dict)
            assert len(result) == 4
            assert result['app1'] == {
                'jobName': 'job1',
                'runType': 'CPU',
                'scaleFactor': 1
            }
            assert result['app2'] == {
                'jobName': 'job1',
                'runType': 'GPU',
                'scaleFactor': 1
            }
            assert result['app3'] == {
                'jobName': 'job2',
                'runType': 'CPU',
                'scaleFactor': 1
            }
            assert result['app4'] == {
                'jobName': 'job2',
                'runType': 'GPU',
                'scaleFactor': 1
            }

    def test_load_qtool_execs(self):
        # Test load_qtool_execs function
        # Create test data
        test_data = pd.DataFrame([
            ['app1', 'sql1', 'node1', True, '', 'Exec1'],
            ['app1', 'sql2', 'node2', False, 'IgnoreNoPerf', 'Exec2'],
            ['app2', 'sql1', 'node1', False, '', 'WholeStageCodegen3'],
            ['app2', 'sql1', 'node2', False, '', 'Exec4']
        ], columns=['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported', 'Action', 'Exec Name'])
        # Call function
        result = load_qtool_execs(test_data)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported']
        assert len(result) == 4
        assert result['Exec Is Supported'].tolist() == [True, True, True, False]

    def test_load_datasets(self):
        """Test load_datasets function"""
        get_config(reload=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dataset structure
            dataset_dir = os.path.join(temp_dir, 'datasets', 'onprem')
            os.makedirs(dataset_dir)

            # Create test JSON file
            test_json = {
                'dataset1': {
                    'eventlogs': ['/path/to/eventlog1'],
                    'platform': 'onprem',
                    'app_meta': {
                        'app1': {
                            'runType': 'CPU',
                            'scaleFactor': 1
                        }
                    }
                }
            }

            json_path = os.path.join(dataset_dir, 'dummy.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(test_json, f)

            # Create dummy preprocessed.parquet cache file
            test_profile = pd.DataFrame([
                ['dataset1', 'app1', 'sql1', 100, 0.8],
                ['dataset1', 'app1', 'sql2', 150, 0.6],
                ['dataset2', 'app2', 'sql1', 200, 0.9],
                ['dataset2', 'app2', 'sql2', 180, 0.7]
            ], columns=['appName', 'appId', 'sqlID', 'Duration', 'fraction_supported'])
            profile_dir = os.path.join(get_config().cache_dir, 'onprem')
            os.makedirs(profile_dir, exist_ok=True)
            test_profile.to_parquet(os.path.join(profile_dir, 'preprocessed.parquet'), index=False)

            # Call function
            datasets, profile_df = load_datasets(dataset_dir)

            # Verify results
            assert isinstance(datasets, dict)
            assert 'dataset1' in datasets
            assert datasets['dataset1']['platform'] == 'onprem'

            assert isinstance(profile_df, pd.DataFrame)
            assert not profile_df.empty
            assert len(profile_df) == 2  # only 2 rows correspond to dataset1
