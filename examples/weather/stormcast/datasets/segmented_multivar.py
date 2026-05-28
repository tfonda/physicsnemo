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


def build_segmented_forecast_minutes(
    *,
    minutes_0_2h: int = 10,
    minutes_2_12h: int = 180,
    minutes_12_24h: int = 720,
) -> list[int]:
    """Build 0-24h segmented lead-time offsets in minutes."""

    if minutes_0_2h <= 0 or minutes_2_12h <= 0 or minutes_12_24h <= 0:
        raise ValueError("All lead-time steps must be positive.")

    offsets = set(range(0, 120 + 1, minutes_0_2h))
    offsets.update(range(120, 720 + 1, minutes_2_12h))
    offsets.update(range(720, 1440 + 1, minutes_12_24h))
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

    lead_time_steps: int = 0

    def __init__(self, params, train):
        super().__init__(params, train)

        self.background_stride_hours = int(params.get("background_stride_hours", 3))
        self.background_stride_minutes = self.background_stride_hours * 60

        self.forecast_minutes = build_segmented_forecast_minutes(
            minutes_0_2h=int(params.get("minutes_0_2h", 10)),
            minutes_2_12h=int(params.get("minutes_2_12h", 180)),
            minutes_12_24h=int(params.get("minutes_12_24h", 720)),
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
        if self.background_stride_minutes <= 60:
            return timestamp

        day_start = datetime(timestamp.year, timestamp.month, timestamp.day)
        minutes_since_day_start = int((timestamp - day_start).total_seconds() // 60)
        aligned_minutes = (
            minutes_since_day_start // self.background_stride_minutes
        ) * self.background_stride_minutes
        return day_start + timedelta(minutes=aligned_minutes)

    def _get_hrrr_field(self, timestamp: datetime) -> np.ndarray:
        ds, _, _ = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, timestamp, timestamp)
        return ds.sel(time=timestamp, channel=self.kept_hrrr_channels).HRRR.values

    def _get_hrrr_target(self, timestamp: datetime) -> np.ndarray:
        if timestamp.minute == 0 and timestamp.second == 0:
            return self._get_hrrr_field(timestamp)

        left_time = timestamp.replace(minute=0, second=0, microsecond=0)
        right_time = left_time + timedelta(hours=1)
        alpha = (timestamp - left_time).total_seconds() / 3600.0

        left = self._get_hrrr_field(left_time)
        right = self._get_hrrr_field(right_time)
        return (1.0 - alpha) * left + alpha * right

    def _split_index(self, global_idx: int) -> tuple[datetime, int]:
        base_idx = global_idx // self.lead_time_steps
        lead_idx = global_idx % self.lead_time_steps
        return self.base_samples[base_idx], lead_idx

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
        hour_aligned = initial_time.replace(minute=0, second=0, microsecond=0)
        lead_minutes = self.forecast_minutes[lead_index]
        ts_tar = hour_aligned + timedelta(minutes=lead_minutes)

        background = self._get_era5(
            self._align_background_timestamp(hour_aligned),
            self._align_background_timestamp(hour_aligned),
        )
        state_inp = self._get_hrrr_field(hour_aligned)
        state_tar = self._get_hrrr_target(ts_tar)
        state_inp, state_tar = self.normalize_state(state_inp), self.normalize_state(state_tar)

        return {
            "background": torch.as_tensor(background),
            "state": (torch.as_tensor(state_inp), torch.as_tensor(state_tar)),
            "lead_time_label": torch.tensor(lead_index, dtype=torch.int64),
        }
