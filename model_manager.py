# model_manager.py - 모델 버전 자동 선택
"""
현재 날짜와 데이터 상황에 따라 최적의 모델을 자동으로 선택합니다.
- 4월 이전: 2025년 모델 사용
- 4월 이후 + 2026년 데이터 충분: 2026년 모델 전환
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        """
        모델 매니저 초기화
        
        Args:
            models_dir: 모델 저장 디렉토리
        """
        self.models_dir = models_dir
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month

    def get_active_model_version(self) -> str:
        """
        현재 날짜와 데이터 기반으로 활성 모델 버전 반환
        
        Returns:
            "2025" 또는 "2026" 등의 모델 버전 문자열
        """
        # 현재 시즌 시작은 3월 20일경
        if self.current_month < 4:
            # 3월 (시즌 초반) → 항상 2025년 모델 사용
            return "2025"
        
        # 4월 이후 → 2026년 모델이 있는지 확인
        if self._model_exists("2026"):
            return "2026"
        else:
            # 2026년 모델이 없으면 2025년 모델 재사용
            return "2025"

    def _model_exists(self, version: str) -> bool:
        """
        특정 버전의 모델이 존재하는지 확인
        
        Args:
            version: 모델 버전 ("2025", "2026" 등)
            
        Returns:
            모델이 존재하면 True
        """
        model_file = os.path.join(self.models_dir, f"model_roll_{version}.pt")
        scaler_file = os.path.join(self.models_dir, f"scaler_roll_{version}.joblib")
        cols_file = os.path.join(self.models_dir, f"feature_cols_roll_{version}.json")

        return all(os.path.exists(f) for f in [model_file, scaler_file, cols_file])

    def get_model_paths(self, version: Optional[str] = None) -> Tuple[str, str, str]:
        """
        모델 파일 경로들을 반환
        
        Args:
            version: 모델 버전 (None이면 활성 모델 자동 선택)
            
        Returns:
            (model_file, scaler_file, cols_file) 경로 튜플
        """
        if version is None:
            version = self.get_active_model_version()

        model_file = os.path.join(self.models_dir, f"model_roll_{version}.pt")
        scaler_file = os.path.join(self.models_dir, f"scaler_roll_{version}.joblib")
        cols_file = os.path.join(self.models_dir, f"feature_cols_roll_{version}.json")

        return model_file, scaler_file, cols_file

    def get_model_metadata(self, version: Optional[str] = None) -> dict:
        """
        모델 메타데이터 반환
        
        Args:
            version: 모델 버전 (None이면 활성 모델 자동 선택)
            
        Returns:
            메타데이터 딕셔너리
        """
        if version is None:
            version = self.get_active_model_version()

        meta_file = os.path.join(self.models_dir, f"model_meta_{version}.json")
        
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "version": version,
                "status": "모델 메타데이터 없음",
                "created_at": "unknown"
            }

    def print_status(self):
        """현재 모델 상태 출력"""
        active_version = self.get_active_model_version()
        metadata = self.get_model_metadata(active_version)

        print("\n" + "="*60)
        print("📊 모델 상태")
        print("="*60)
        print(f"📅 현재 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔴 활성 모델: {active_version}년")
        print(f"📋 메타데이터:")
        for key, val in metadata.items():
            if key != "version":
                print(f"   - {key}: {val}")
        
        # 다른 버전도 확인
        if self._model_exists("2025") and active_version != "2025":
            print(f"\n💾 대체 모델 가능: 2025년")
        if self._model_exists("2026") and active_version != "2026":
            print(f"💾 대체 모델 가능: 2026년")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    manager = ModelManager()
    manager.print_status()
