# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset adapter for PhysicalAI compatibility."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning_utilities import module_available

from physicalai.data.dataset import Dataset
from physicalai.data.observation import Feature, FeatureType, NormalizationParameters, Observation

from .converters import FormatConverter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from physicalai.data import Observation

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

logger = logging.getLogger(__name__)

_DECODE_BACKEND_FALLBACKS: dict[str | None, str] = {
    None: "pyav",
    "pyav": "torchcodec",
    "torchcodec": "pyav",
}


def _is_video_decode_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "avcodec_send_packet()",
            "decodeAVFrame",
            "Could not push packet to decoder",
            "Function not implemented",
        )
    )


class _LeRobotDatasetAdapter(Dataset):
    """An internal adapter that makes a `LeRobotDataset` compatible with the `physicalai.data.Dataset` interface.

    This adapter class serves two primary purposes:
    1.  **Protocol Compliance**: It wraps the `LeRobotDataset` to ensure it conforms to the
        abstract methods and properties required by the `physicalai.data.Dataset` base class
        (e.g., providing `.features`, `.fps`, etc.).
    2.  **Interface Adaptation**: It transforms the dictionary-based output of `LeRobotDataset.__getitem__`
        into the structured `Observation` dataclass format expected by the training pipeline.

    Note:
        This is an internal implementation detail and is not meant to be used directly by end-users.
        The `LeRobotDataModule` handles the creation and management of this adapter automatically.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> None:
        """Initialize a _LeRobotDatasetAdapter.

        This adapter initializes an internal `LeRobotDataset` using the provided configuration
        and exposes the same dataset interface for action training.

        Args:
            repo_id (str): Repository ID of the LeRobot dataset.
            root (str | Path | None, optional): Local root directory to cache dataset files.
                Defaults to `None`.
            episodes (list[int] | None, optional): Specific episode indices to include.
                Defaults to `None`.
            image_transforms (Callable | None, optional): Transformations to apply to images.
                Defaults to `None`.
            delta_timestamps (dict[str, list[float]] | None, optional): Mapping of signal keys to timestamp offsets.
                Defaults to `None`.
            tolerance_s (float, optional): Tolerance in seconds when aligning timestamps.
                Defaults to `1e-4`.
            revision (str | None, optional): Dataset version or branch to use.
                Defaults to `None`.
            force_cache_sync (bool, optional): If True, forces synchronization of the dataset cache.
                Defaults to `False`.
            download_videos (bool, optional): Whether to download associated videos.
                Defaults to `True`.
            video_backend (str | None, optional): Backend to use for video decoding.
                Defaults to `None`.
            batch_encoding_size (int, optional): Number of samples per encoded batch.
                Defaults to `1`.

        Raises:
            ImportError: If `lerobot` is not installed.
        """
        super().__init__()

        if LeRobotDataset is None:
            msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
            raise ImportError(msg)

        self._lerobot_kwargs = {
            "repo_id": repo_id,
            "root": root,
            "episodes": episodes,
            "image_transforms": image_transforms,
            "delta_timestamps": delta_timestamps,
            "tolerance_s": tolerance_s,
            "revision": revision,
            "force_cache_sync": force_cache_sync,
            "download_videos": download_videos,
            "batch_encoding_size": batch_encoding_size,
        }
        self._video_backend = video_backend
        self._lerobot_dataset = self._build_lerobot_dataset(video_backend)

    def _build_lerobot_dataset(self, video_backend: str | None) -> LeRobotDataset:
        return LeRobotDataset(
            **self._lerobot_kwargs,
            video_backend=video_backend,
        )

    def _repair_corrupt_videos_for_index(self, idx: int) -> None:
        from .utils.video_repair import find_videos_with_corrupt_frame, repair_corrupt_video

        corrupt_videos = find_videos_with_corrupt_frame(self._lerobot_dataset, idx)
        for video_path in corrupt_videos:
            logger.warning("Repairing corrupt training video %s for sample %s", video_path, idx)
            repair_corrupt_video(video_path)

    def _retry_with_fallback_backend(self, idx: int, exc: Exception) -> Observation:
        original_backend = self._video_backend
        fallback_backend = _DECODE_BACKEND_FALLBACKS.get(original_backend)
        if fallback_backend is None or not hasattr(self, "_lerobot_kwargs"):
            raise exc

        logger.warning(
            "Video decode failed with backend %r; retrying sample %s with %r",
            original_backend,
            idx,
            fallback_backend,
        )
        fallback_dataset = self._build_lerobot_dataset(fallback_backend)
        try:
            observation = FormatConverter.to_observation(fallback_dataset[idx])
        except Exception as fallback_exc:
            if not _is_video_decode_error(fallback_exc):
                raise
            logger.warning(
                "Video decode still failed for sample %s after backend fallback; attempting targeted repair",
                idx,
                exc_info=True,
            )
            self._repair_corrupt_videos_for_index(idx)

            repaired_dataset = self._build_lerobot_dataset(original_backend)
            repaired_backend = original_backend
            try:
                observation = FormatConverter.to_observation(repaired_dataset[idx])
            except Exception:
                repaired_dataset = self._build_lerobot_dataset(fallback_backend)
                repaired_backend = fallback_backend
                observation = FormatConverter.to_observation(repaired_dataset[idx])

            self._lerobot_dataset = repaired_dataset
            self._video_backend = repaired_backend
            return observation

        self._lerobot_dataset = fallback_dataset
        self._video_backend = fallback_backend
        return observation

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._lerobot_dataset)

    def __getitem__(self, idx: int) -> Observation:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Observation: The item from the dataset.
        """
        try:
            return FormatConverter.to_observation(self._lerobot_dataset[idx])
        except Exception as exc:
            if not _is_video_decode_error(exc):
                raise
            return self._retry_with_fallback_backend(idx, exc)

    @staticmethod
    def from_lerobot(lerobot_dataset: LeRobotDataset) -> _LeRobotDatasetAdapter:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.

        Returns:
            _LeRobotDatasetAdapter: A new adapter instance that uses the provided dataset.
        """
        instance = _LeRobotDatasetAdapter.__new__(_LeRobotDatasetAdapter)
        # Bypassing __init__ to set the internal dataset
        instance._lerobot_dataset = lerobot_dataset  # noqa: SLF001
        instance._video_backend = None  # noqa: SLF001
        return instance

    @property
    def raw_features(self) -> dict[str, dict[Any, Any]]:
        """Raw dataset features."""
        return self._lerobot_dataset.features

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Observation features from the dataset."""
        dataset_features = self._lerobot_dataset.features
        raw_obs_features = {key: ft for key, ft in dataset_features.items() if key.startswith("observation")}
        dataset_meta = self._lerobot_dataset.meta

        observation_features = {}
        for k in raw_obs_features:
            if k in dataset_meta.features:
                feature_name = k[len("observation.") :]  # Remove "observation." prefix, filtering was done above
                feature_type = FeatureType.STATE
                feature_shape = dataset_meta.features[k]["shape"]
                if dataset_meta.features[k]["dtype"] in {"image", "video"}:
                    feature_type = FeatureType.VISUAL
                    # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0
                    # for ported datasets.
                    if "images." in feature_name:
                        feature_name = feature_name[len("images.") :]
                    if dataset_meta.features[k]["names"][2] in {"channel", "channels"}:  # (h, w, c) -> (c, h, w)
                        feature_shape = (feature_shape[2], feature_shape[0], feature_shape[1])
                elif k == "observation.environment_state":
                    feature_type = FeatureType.ENV

                stats = dataset_meta.stats[k]
                observation_features[feature_name] = Feature(
                    ftype=feature_type,
                    normalization_data=NormalizationParameters(
                        mean=stats["mean"].tolist(),
                        std=stats["std"].tolist(),
                        min=stats["min"].tolist(),
                        max=stats["max"].tolist(),
                        q01=stats["q01"].tolist() if "q01" in stats else None,
                        q99=stats["q99"].tolist() if "q99" in stats else None,
                    ),
                    shape=feature_shape,
                    name=feature_name,
                )
        return observation_features

    @property
    def action_features(self) -> dict[str, Feature]:
        """Action features from LeRobot dataset."""
        dataset_features = self._lerobot_dataset.features
        raw_act_features = {key: ft for key, ft in dataset_features.items() if key.startswith("action")}
        dataset_meta = self._lerobot_dataset.meta

        action_features = {}
        for k in raw_act_features:
            if k in dataset_meta.features:
                stats = dataset_meta.stats[k]
                action_features[k] = Feature(
                    ftype=FeatureType.ACTION,
                    normalization_data=NormalizationParameters(
                        mean=stats["mean"].tolist(),
                        std=stats["std"].tolist(),
                        min=stats["min"].tolist(),
                        max=stats["max"].tolist(),
                        q01=stats["q01"].tolist() if "q01" in stats else None,
                        q99=stats["q99"].tolist() if "q99" in stats else None,
                    ),
                    shape=dataset_meta.features[k]["shape"],
                    name=k,
                )

        return action_features

    @property
    def fps(self) -> int:
        """Frames per second of dataset."""
        return self._lerobot_dataset.fps

    @property
    def tolerance_s(self) -> float:
        """Tolerance to keep delta timestamps in sync with fps."""
        return self._lerobot_dataset.tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Expose delta_indices from the DatasetReader (lerobot >=0.5.1)."""
        reader = getattr(self._lerobot_dataset, "reader", None)
        if reader is not None:
            return reader.delta_indices or {}
        return getattr(self._lerobot_dataset, "delta_indices", None) or {}

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Set delta_indices on DatasetReader so the training callback can inject them post-construction."""
        reader = getattr(self._lerobot_dataset, "reader", None)
        if reader is not None:
            reader.delta_indices = indices
        else:
            self._lerobot_dataset.delta_indices = indices


__all__ = ["_LeRobotDatasetAdapter"]
