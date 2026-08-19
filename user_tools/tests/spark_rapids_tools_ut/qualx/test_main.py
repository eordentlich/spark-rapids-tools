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
    _build_duration_evaluation_results,
    _get_model_path,
    _get_model,
    _compute_summary,
    _merge_qxs_app_predictions,
    _merge_qxs_sql_predictions,
    _read_dataset_scores,
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
            'split': ['train', 'train'],
        })

        rolled = _roll_up_stage_type_predictions(results)
        assert len(rolled) == 1
        assert rolled['duration_sum'].iloc[0] == 400
        assert rolled['duration_sum_pred'].iloc[0] == 150
        assert rolled['speedup_pred'].iloc[0] == 400 / 150
        assert rolled['split'].iloc[0] == 'train'

        summary = _compute_summary(results)
        assert summary['duration_sum'].iloc[0] == 400
        assert summary['duration_sum_pred'].iloc[0] == 150
        assert summary['appDuration_pred'].iloc[0] == 750
        assert summary['speedup'].iloc[0] == 1000 / 750

        monkeypatch.setenv('QUALX_LABEL', 'Duration')
        monkeypatch.setenv('QUALX_DURATION_SUM_STAGE_TYPE', 'false')
        get_config(reload=True)

    def test_roll_up_stage_type_predictions_rejects_inconsistent_splits(self, monkeypatch):
        """StageType rows for one SQL cannot cross model splits."""
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
            'split': ['train', 'val'],
        })

        with pytest.raises(ValueError, match='inconsistent split assignments'):
            _roll_up_stage_type_predictions(results)

        monkeypatch.setenv('QUALX_LABEL', 'Duration')
        monkeypatch.setenv('QUALX_DURATION_SUM_STAGE_TYPE', 'false')
        get_config(reload=True)

    def test_merge_qxs_sql_predictions_uses_identity_and_fallback(self):
        """Filtered duration differences should not prevent QXS SQL predictions from joining."""
        raw_sql = pd.DataFrame({
            'appId': ['id1', 'id2'],
            'sqlID': [1, 2],
            'scaleFactor': [1.0, 1.0],
            'appDuration': [100, 80],
            'duration_sum': [100, 80],
        })
        filtered_sql = pd.DataFrame({
            'appId': ['id1'],
            'sqlID': [1],
            'scaleFactor': [1.0],
            'appDuration': [90],
            'duration_sum': [90],
            'duration_sum_supported': [45],
            'duration_sum_pred': [60],
        })

        result = _merge_qxs_sql_predictions(raw_sql, filtered_sql, 'duration_sum')

        assert result['_qxs_duration_sum_supported'].tolist() == [45, 0]
        assert result['_qxs_duration_sum_pred'].tolist() == [70, 80]
        assert result['_qxs_speedup'].tolist() == pytest.approx([100 / 70, 1.0])
        assert not result.filter(like='_qxs_').isna().any().any()

    def test_build_stage_type_evaluation_results(self):
        """StageType evaluation should join QXS rows within the corresponding bin."""
        raw_stage_type = pd.DataFrame({
            'appId': ['id1', 'id1'],
            'sqlID': [1, 1],
            'scaleFactor': [1.0, 1.0],
            'appDuration': [400, 400],
            STAGE_TYPE_COL: [STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN],
            'duration_sum': [100, 300],
            'gpu_duration_sum': [50, 100],
            'y': [2.0, 3.0],
            'duration_sum_supported': [100, 300],
            'duration_sum_pred': [50, 100],
            'y_pred': [2.0, 3.0],
            'split': ['test', 'test'],
        })
        filtered_stage_type = pd.DataFrame({
            'appId': ['id1'],
            'sqlID': [1],
            'scaleFactor': [1.0],
            STAGE_TYPE_COL: [STAGE_TYPE_INPUT_SCAN],
            'duration_sum': [90],
            'duration_sum_supported': [45],
            'duration_sum_pred': [60],
        })

        result = _build_duration_evaluation_results(
            raw_stage_type,
            filtered_stage_type,
            'duration_sum',
            stage_type=True,
        )

        assert result[STAGE_TYPE_COL].tolist() == [
            STAGE_TYPE_INPUT_SCAN,
            STAGE_TYPE_NO_INPUT_SCAN,
        ]
        assert result['Actual speedup'].tolist() == [2.0, 3.0]
        assert result['QX speedup'].tolist() == [2.0, 3.0]
        assert result['QXS duration_sum_pred'].tolist() == [70, 300]
        assert result['QXS speedup'].tolist() == pytest.approx([100 / 70, 1.0])

    def test_read_dataset_scores_allows_empty_stage_type(self, tmp_path):
        """The nullable stageType identifier must not invalidate SQL score rows."""
        scores = pd.DataFrame({
            'model': ['model1', 'model1'],
            'platform': ['platform1', 'platform1'],
            'dataset': ['dataset1', 'dataset1'],
            'granularity': ['sql', 'stageType'],
            STAGE_TYPE_COL: [pd.NA, STAGE_TYPE_INPUT_SCAN],
            'split': ['all', 'all'],
            'score': ['MAPE', 'MAPE'],
            'QX': [0.1, 0.2],
            'QXS': [0.3, 0.4],
        })
        scores.to_csv(tmp_path / 'dataset1_mape.csv', index=False)

        sql_scores = _read_dataset_scores(str(tmp_path), 'MAPE', 'sql', 'all')
        stage_type_scores = _read_dataset_scores(
            str(tmp_path),
            'MAPE',
            'stageType',
            'all',
        )

        assert len(sql_scores) == 1
        assert sql_scores['QX'].iloc[0] == 0.1
        assert len(stage_type_scores) == 1
        assert stage_type_scores[STAGE_TYPE_COL].iloc[0] == STAGE_TYPE_INPUT_SCAN

    def test_merge_qxs_app_predictions_uses_identity_and_fallback(self):
        """Filtered duration differences should be treated as unaccelerated application time."""
        raw_app = pd.DataFrame({
            'appId': ['id1', 'id2'],
            'appDuration': [100, 200],
        })
        filtered_app = pd.DataFrame({
            'appId': ['id1'],
            'appDuration': [90],
            'duration_sum_pred': [60],
            'duration_sum_supported': [45],
            'appDuration_pred': [70],
        })

        result = _merge_qxs_app_predictions(raw_app, filtered_app, 'duration_sum')

        assert result['_qxs_duration_sum_pred'].tolist() == [70, 0]
        assert result['_qxs_duration_sum_supported'].tolist() == [45, 0]
        assert result['_qxs_fraction_supported'].tolist() == pytest.approx([0.45, 0.0])
        assert result['_qxs_appDuration_pred'].tolist() == [80, 200]
        assert result['_qxs_speedup'].tolist() == pytest.approx([1.25, 1.0])
        assert not result.filter(like='_qxs_').isna().any().any()
