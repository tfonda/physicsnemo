# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from datetime import datetime, timedelta

import numpy as np
import torch

from .data_loader_hrrr_era5 import HrrrEra5Dataset

MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600.0


def build_segmented_forecast_minutes(
    *,
    step_minutes_0_2h: int = 10,
    step_minutes_2_12h: int = 180,
    step_minutes_12_24h: int = 720,
) -> list[int]:
    """Build 0-24h segmented lead-time offsets in minutes."""

    if step_minutes_0_2h <= 0 or step_minutes_2_12h <= 0 or step_minutes_12_24h <= 0:
        raise ValueError("All lead-time steps must be positive.")

    offsets = set(range(0, 120 + 1, step_minutes_0_2h))
    offsets.update(range(120, 720 + 1, step_minutes_2_12h))
    offsets.update(range(720, 1440 + 1, step_minutes_12_24h))
    return sorted(offsets)


class SegmentedMultivarHrrrEra5Dataset(HrrrEra5Dataset):
    """HRRR/ERA5 dataset with segmented lead times and multivariate state targets.

    This recipe targets joint precipitation and 2m temperature forecasting with:
    - 0-2h every 10 minutes
    - 2-12h every 3 hours
    - 12-24h every 12 hours

    For lead times that are not integer hours, target states are linearly interpolated
    from adjacent hourly state snapshots.
    """

    def __init__(self, params, train):
        super().__init__(params, train)

        def _int_param(name: str, default: int) -> int:
            value = params.get(name, default)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Dataset parameter '{name}' must be an integer.") from exc

        self.background_stride_hours = _int_param("background_stride_hours", 3)
        self.background_stride_minutes = self.background_stride_hours * MINUTES_PER_HOUR

        self.forecast_minutes = build_segmented_forecast_minutes(
            step_minutes_0_2h=_int_param("minutes_0_2h", 10),
            step_minutes_2_12h=_int_param("minutes_2_12h", 180),
            step_minutes_12_24h=_int_param("minutes_12_24h", 720),
        )
        self.forecast_offsets_hours = [minute / 60.0 for minute in self.forecast_minutes]
        self.lead_time_steps = len(self.forecast_minutes)
        self._max_forecast_minutes = max(self.forecast_minutes)

        valid_sample_set = set(self.valid_samples)
        self.base_samples = [
            ts
            for ts in self.valid_samples
            if (ts + timedelta(minutes=self._max_forecast_minutes)) in valid_sample_set
        ]

    def __len__(self):
        return len(self.base_samples) * self.lead_time_steps

    def _align_background_timestamp(self, timestamp: datetime) -> datetime:
        """Align timestamp to the latest coarse-background cycle."""
        if self.background_stride_minutes <= MINUTES_PER_HOUR:
            return timestamp

        day_start = datetime(timestamp.year, timestamp.month, timestamp.day)
        minutes_since_day_start = int((timestamp - day_start).total_seconds() // 60)
        aligned_minutes = (
            minutes_since_day_start // self.background_stride_minutes
        ) * self.background_stride_minutes
        return day_start + timedelta(minutes=aligned_minutes)

    def _get_hrrr_field(self, timestamp: datetime) -> np.ndarray:
        """Load one exact-timestamp HRRR state field."""
        # Pass timestamp twice because _get_ds_handles expects an input/target pair.
        # It returns (input_ds, target_ds, is_adjacent_year); for an exact timestamp
        # fetch we only need the first dataset handle.
        ds, _, _ = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, timestamp, timestamp)
        return ds.sel(time=timestamp, channel=self.kept_hrrr_channels).HRRR.values

    def _get_hrrr_target(self, timestamp: datetime) -> np.ndarray:
        """Load one target field, linearly interpolating when timestamp is sub-hourly."""
        if timestamp.minute == 0 and timestamp.second == 0:
            return self._get_hrrr_field(timestamp)

        left_time = timestamp.replace(minute=0, second=0, microsecond=0)
        right_time = left_time + timedelta(hours=1)
        alpha = (timestamp - left_time).total_seconds() / SECONDS_PER_HOUR

        left = self._get_hrrr_field(left_time)
        right = self._get_hrrr_field(right_time)
        return (1.0 - alpha) * left + alpha * right

    def _split_index(self, global_idx: int) -> tuple[datetime, int]:
        """Convert a global item index into (base timestamp, lead-time index)."""
        sample_idx = global_idx // self.lead_time_steps
        lead_idx = global_idx % self.lead_time_steps
        return self.base_samples[sample_idx], lead_idx

    def __getitem__(self, global_idx):
        ts_inp, lead_idx = self._split_index(global_idx)
        lead_minutes = self.forecast_minutes[lead_idx]
        ts_tar = ts_inp + timedelta(minutes=lead_minutes)

        bg_timestamp = self._align_background_timestamp(ts_inp)
        background = self._get_era5(bg_timestamp, bg_timestamp)

        state_inp = self._get_hrrr_field(ts_inp)
        state_tar = self._get_hrrr_target(ts_tar)
        state_inp, state_tar = self.normalize_state(state_inp), self.normalize_state(state_tar)

        return {
            "background": background,
            "state": (torch.as_tensor(state_inp), torch.as_tensor(state_tar)),
            "lead_time_label": torch.tensor(lead_idx, dtype=torch.int64),
        }

    def get_forecast(self, initial_time: datetime, lead_index: int):
        """Create one inference sample for a given initial time and lead index."""
        hour_aligned = initial_time.replace(minute=0, second=0, microsecond=0)
        lead_minutes = self.forecast_minutes[lead_index]
        ts_tar = hour_aligned + timedelta(minutes=lead_minutes)
        bg_timestamp = self._align_background_timestamp(hour_aligned)

        # _get_era5 currently consumes only its first timestamp argument. We keep
        # both arguments equal for compatibility with the parent method signature.
        background = self._get_era5(
            bg_timestamp,
            bg_timestamp,
        )
        state_inp = self._get_hrrr_field(hour_aligned)
        state_tar = self._get_hrrr_target(ts_tar)
        state_inp, state_tar = self.normalize_state(state_inp), self.normalize_state(state_tar)

        return {
            "background": torch.as_tensor(background),
            "state": (torch.as_tensor(state_inp), torch.as_tensor(state_tar)),
            "lead_time_label": torch.tensor(lead_index, dtype=torch.int64),
        }
