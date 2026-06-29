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

"""Test qualx_main module"""

import pytest  # pylint: disable=import-error
import pandas as pd

from spark_rapids_tools.tools.qualx.qualx_main import (
    _get_model_path,
    _get_model,
    _compute_summary,
    _roll_up_stage_type_predictions,
    _add_entries_for_missing_apps,
)
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.stage_type import (
    STAGE_TYPE_COL,
    STAGE_TYPE_INPUT_SCAN,
    STAGE_TYPE_NO_INPUT_SCAN,
)
from ..conftest import SparkRapidsToolsUT


class TestMain(SparkRapidsToolsUT):
    """Test class for qualx_main module"""

    def test_get_model_path_with_platform(self):
        """Test _get_model_path with platform only"""
        platform = 'onprem'
        model_path = _get_model_path(platform, None)
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_with_model(self):
        '''Test _get_model_path with explicit model name'''
        platform = 'onprem'
        model = 'onprem'
        model_path = _get_model_path(platform, model)
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_with_variant(self):
        '''Test _get_model_path with platform variant'''
        platform = 'onprem'
        variant = 'PHOTON'
        model_path = _get_model_path(platform, None, variant)
        # Should fall back to base platform model if variant model doesn't exist
        assert model_path.name == 'onprem.json'
        assert model_path.exists()

    def test_get_model_path_invalid_platform(self):
        '''Test _get_model_path with invalid platform'''
        platform = 'invalid_platform'
        with pytest.raises(ValueError, match=r'Platform \[invalid_platform\] does not have a pre-trained model'):
            _get_model_path(platform, None)

    def test_get_model_with_platform(self):
        '''Test _get_model with platform only'''
        platform = 'onprem'
        model = _get_model(platform)
        assert model is not None
        assert hasattr(model, 'predict')

    def test_compute_summary(self):
        """Test _compute_summary with sample data"""
        # Create sample input data
        data = {
            'appName': ['app1', 'app1', 'app2'],
            'appId': ['id1', 'id1', 'id2'],
            'appDuration': [100, 100, 200],
            'Duration': [100, 100, 200],
            'Duration_pred': [50, 50, 100],
            'Duration_supported': [100, 100, 200],
            'sqlDuration': [50, 50, 100],
            'sqlDuration_pred': [25, 25, 50],
            'sqlDuration_supported': [50, 50, 100],
            'description': ['desc1', 'desc1', 'desc2'],
            'scaleFactor': [1.0, 1.0, 1.0]
        }
        results = pd.DataFrame(data)

        summary = _compute_summary(results)

        # Verify summary calculations
        assert len(summary) == 2  # Should have 2 unique apps
        assert all(col in summary.columns for col in [
            'appName', 'appId', 'appDuration', 'fraction_supported',
            'appDuration_pred', 'speedup'
        ])

    def test_add_entries_for_missing_apps(self):
        """Test _add_entries_for_missing_apps"""
        # Create sample data
        all_default_preds = pd.DataFrame({
            'appId': ['id1', 'id2', 'id3'],
            'appName': ['app1', 'app2', 'app3'],
            'appDuration': [100, 200, 300]
        })

        summary_df = pd.DataFrame({
            'appId': ['id1', 'id2'],
            'appName': ['app1', 'app2'],
            'appDuration': [100, 200],
            'speedup': [2.0, 1.5]
        })

        result = _add_entries_for_missing_apps(all_default_preds, summary_df)

        # Verify results
        assert len(result) == 3  # Should include all apps
        assert 'wasPredicted' in result.columns
        assert sum(result['wasPredicted']) == 2  # Two apps were predicted
        assert not result.loc[result['appId'] == 'id3', 'wasPredicted'].iloc[0]  # id3 was not predicted

    def test_roll_up_stage_type_predictions(self, monkeypatch):
        """Test stageType rows roll up using summed predicted durations."""
        monkeypatch.setenv('QUALX_LABEL', 'duration_sum')
        monkeypatch.setenv('QUALX_DURATION_SUM_STAGE_TYPE', 'true')
        get_config(reload=True)

        results = pd.DataFrame({
            'appName': ['app1', 'app1'],
            'appId': ['id1', 'id1'],
            'appDuration': [1000, 1000],
            'sqlID': [7, 7],
            'scaleFactor': [1.0, 1.0],
            'description': ['query', 'query'],
            STAGE_TYPE_COL: [STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN],
            'duration_sum': [100, 300],
            'duration_sum_pred': [50, 100],
            'duration_sum_supported': [100, 300],
            'y_pred': [2.0, 3.0],
        })

        rolled = _roll_up_stage_type_predictions(results)
        assert len(rolled) == 1
        assert rolled['duration_sum'].iloc[0] == 400
        assert rolled['duration_sum_pred'].iloc[0] == 150
        assert rolled['speedup_pred'].iloc[0] == 400 / 150

        summary = _compute_summary(results)
        assert summary['duration_sum'].iloc[0] == 400
        assert summary['duration_sum_pred'].iloc[0] == 150
        assert summary['appDuration_pred'].iloc[0] == 750
        assert summary['speedup'].iloc[0] == 1000 / 750

        monkeypatch.setenv('QUALX_LABEL', 'Duration')
        monkeypatch.setenv('QUALX_DURATION_SUM_STAGE_TYPE', 'false')
        get_config(reload=True)
