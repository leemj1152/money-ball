from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Tuple


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month

    def _versioned_paths(self, version: str) -> Tuple[str, str, str]:
        return (
            os.path.join(self.models_dir, f"model_roll_{version}.pt"),
            os.path.join(self.models_dir, f"scaler_roll_{version}.joblib"),
            os.path.join(self.models_dir, f"feature_cols_roll_{version}.json"),
        )

    def _legacy_paths(self) -> Tuple[str, str, str]:
        return (
            os.path.join(self.models_dir, "model_roll.pt"),
            os.path.join(self.models_dir, "scaler_roll.joblib"),
            os.path.join(self.models_dir, "feature_cols_roll.json"),
        )

    def _paths_exist(self, paths: Tuple[str, str, str]) -> bool:
        return all(os.path.exists(path) for path in paths)

    def _model_exists(self, version: str) -> bool:
        return self._paths_exist(self._versioned_paths(version))

    def _discover_available_versions(self) -> list[str]:
        versions: list[str] = []
        if not os.path.isdir(self.models_dir):
            return versions

        prefix = "model_roll_"
        suffix = ".pt"
        for name in os.listdir(self.models_dir):
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            version = name[len(prefix):-len(suffix)]
            if self._model_exists(version):
                versions.append(version)
        return sorted(versions)

    def get_active_model_version(self) -> str:
        preferred_version = str(self.current_year - 1 if self.current_month < 4 else self.current_year)
        if self._model_exists(preferred_version):
            return preferred_version

        available_versions = self._discover_available_versions()
        if available_versions:
            return available_versions[-1]

        if self._paths_exist(self._legacy_paths()):
            return "legacy"

        raise FileNotFoundError(
            "No rolling model files were found. Expected either versioned files like "
            "models/model_roll_2025.pt + scaler + feature cols, or legacy files "
            "models/model_roll.pt + scaler_roll.joblib + feature_cols_roll.json."
        )

    def get_model_paths(self, version: Optional[str] = None) -> Tuple[str, str, str]:
        if version is None:
            version = self.get_active_model_version()

        if version == "legacy":
            paths = self._legacy_paths()
            if self._paths_exist(paths):
                return paths
        else:
            versioned = self._versioned_paths(version)
            if self._paths_exist(versioned):
                return versioned

        if self._paths_exist(self._legacy_paths()):
            return self._legacy_paths()

        available_versions = self._discover_available_versions()
        if available_versions:
            fallback = self._versioned_paths(available_versions[-1])
            if self._paths_exist(fallback):
                return fallback

        raise FileNotFoundError(f"No usable rolling model found for version={version!r}.")

    def get_model_metadata(self, version: Optional[str] = None) -> dict:
        if version is None:
            version = self.get_active_model_version()

        meta_candidates = []
        if version == "legacy":
            meta_candidates.append(os.path.join(self.models_dir, "model_meta.json"))
            meta_candidates.append(os.path.join(self.models_dir, "model_meta_roll.json"))
        else:
            meta_candidates.append(os.path.join(self.models_dir, f"model_meta_{version}.json"))

        for meta_file in meta_candidates:
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)

        return {
            "version": version,
            "status": "metadata_not_found",
            "created_at": "unknown",
        }

    def print_status(self):
        active_version = self.get_active_model_version()
        metadata = self.get_model_metadata(active_version)
        print("=" * 60)
        print("Model status")
        print(f"Current time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Active model: {active_version}")
        for key, val in metadata.items():
            if key != "version":
                print(f"  - {key}: {val}")
        print("=" * 60)


if __name__ == "__main__":
    ModelManager().print_status()
