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

"""Stage-type feature constants for duration_sum QualX models."""

STAGE_TYPE_COL = 'stageType'
STAGE_TYPE_INPUT_SCAN = 0
STAGE_TYPE_NO_INPUT_SCAN = 1
STAGE_TYPE_VALUES = (STAGE_TYPE_INPUT_SCAN, STAGE_TYPE_NO_INPUT_SCAN)

# Input-scan stages are identified by SQL plan nodes such as "Scan parquet" and "GpuScan parquet".
# The negative look-behind avoids treating names such as "LocalTableScan" as input scans.
SCAN_NODE_PATTERN = r'(?<![A-Za-z])(?:Gpu)?Scan\b'
