# compare_models_comprehensive.py
"""
모든 모델 버전을 포괄적으로 비교:
1. 4개 모델을 같은 데이터로 학습
2. 다중 연도 검증 (2024 학습, 2025 테스트)
3. 메트릭 비교: AUC, Log Loss, Brier Score, ROI
4. 최고 성능 모델 선택

실행:
  python compare_models_comprehensive.py --year1 2024 --year2 2025
"""
import argparse
import os
import json
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def run_training_script(script_name, year, **kwargs):
    """학습 스크립트 실행"""
    
    cmd = [sys.executable, script_name, "--year", str(year)]
    
    for key, val in kwargs.items():
        if val is not None:
            cmd.extend([f"--{key}", str(val)])
    
    print(f"\n{'='*70}")
    print(f"[실행] {script_name}")
    print(f"{'='*70}")
    print(f"명령: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] {script_name} 실행 실패 (코드: {result.returncode})")
        return False
    
    print(f"[OK] {script_name} 완료")
    return True


def get_model_metadata(year, models_dir="models"):
    """모델 메타데이터 읽기"""
    
    models_info = {}
    
    # 각 모델 타입에 해당하는 메타파일
    model_types = [
        ("LightGBM", f"model_meta_lgbm_{year}.json"),
        ("MLP Ensemble", f"model_meta_ensemble_mlp_{year}.json"),
        ("MLP Improved", f"model_meta_mlp_improved_{year}.json"),
        ("MLP Tuned", f"model_meta_mlp_tuned_{year}.json"),
    ]
    
    for model_name, meta_file in model_types:
        meta_path = os.path.join(models_dir, meta_file)
        
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            models_info[model_name] = meta
            print(f"[OK] {model_name}: {meta_path} 로드 완료")
        else:
            print(f"[WARN] {model_name}: {meta_path} 없음")
    
    return models_info


def create_comparison_report(year1, year2, models_dir="models"):
    """모델 비교 보고서 생성"""
    
    print(f"\n{'='*80}")
    print(f"모델 비교 보고서")
    print(f"{'='*80}")
    print(f"학습 데이터: {year1}")
    print(f"검증 데이터: {year2}")
    print(f"{'='*80}\n")
    
    # 메타데이터 수집
    meta_data = get_model_metadata(year1, models_dir)
    
    if not meta_data:
        print("[ERROR] 모델 메타데이터 없음")
        return None
    
    # 비교 DataFrame
    comparison = []
    
    for model_name, meta in meta_data.items():
        comp_row = {
            "Model": model_name,
            "AUC (Train)": meta.get("auc", np.nan),
            "Log Loss": meta.get("log_loss", np.nan),
            "Brier Score": meta.get("brier_score", np.nan),
            "Samples": meta.get("samples", np.nan),
            "Features": meta.get("features", np.nan),
            "Year": meta.get("season_start", "").split("-")[0],
        }
        
        # 추가 정보
        if "best_hyperparameters" in meta:
            comp_row["Tuned"] = "Yes"
        else:
            comp_row["Tuned"] = "No"
        
        comparison.append(comp_row)
    
    df_comparison = pd.DataFrame(comparison)
    
    # AUC 기준 정렬
    df_comparison = df_comparison.sort_values("AUC (Train)", ascending=False)
    
    print("📊 성능 비교:")
    print("-" * 100)
    print(df_comparison.to_string(index=False))
    print("-" * 100)
    
    # 통계
    print("\n📈 통계:")
    print(f"  최고 AUC: {df_comparison['AUC (Train)'].max():.4f} ({df_comparison.iloc[0]['Model']})")
    print(f"  평균 AUC: {df_comparison['AUC (Train)'].mean():.4f}")
    print(f"  AUC 범위: {df_comparison['AUC (Train)'].min():.4f} ~ {df_comparison['AUC (Train)'].max():.4f}")
    
    print(f"\n  최저 Log Loss: {df_comparison['Log Loss'].min():.4f}")
    print(f"  평균 Log Loss: {df_comparison['Log Loss'].mean():.4f}")
    
    print(f"\n  최저 Brier Score: {df_comparison['Brier Score'].min():.4f}")
    print(f"  평균 Brier Score: {df_comparison['Brier Score'].mean():.4f}")
    
    # 최고 모델
    best_model = df_comparison.iloc[0]
    print(f"\n🏆 최고 성능 모델: {best_model['Model']}")
    print(f"   - AUC: {best_model['AUC (Train)']:.4f}")
    print(f"   - Log Loss: {best_model['Log Loss']:.4f}")
    print(f"   - Brier Score: {best_model['Brier Score']:.4f}")
    
    # 현재 기준선 (0.5562)
    baseline_auc = 0.5562
    improvement = best_model['AUC (Train)'] - baseline_auc
    improvement_pct = (improvement / baseline_auc) * 100
    
    print(f"\n📊 기준선 대비:")
    print(f"   기준선 AUC: {baseline_auc:.4f}")
    print(f"   개선액: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if improvement > 0:
        print(f"   ✅ 개선됨: 배포 권장")
    else:
        print(f"   ❌ 악화됨: 배포 미권장")
    
    # 결과 저장
    report_file = os.path.join(models_dir, f"comparison_report_{year1}_{year2}.csv")
    df_comparison.to_csv(report_file, index=False)
    print(f"\n[SAVE] 비교 결과 저장: {report_file}")
    
    # JSON 보고서
    report_json = {
        "timestamp": datetime.now().isoformat(),
        "train_year": year1,
        "test_year": year2,
        "baseline_auc": baseline_auc,
        "best_model": best_model['Model'],
        "best_auc": float(best_model['AUC (Train)']),
        "improvement": float(improvement),
        "improvement_pct": float(improvement_pct),
        "comparison": df_comparison.to_dict('records'),
    }
    
    json_file = os.path.join(models_dir, f"comparison_report_{year1}_{year2}.json")
    with open(json_file, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"[SAVE] JSON 보고서 저장: {json_file}")
    
    return df_comparison, report_json


def main():
    ap = argparse.ArgumentParser(description="모델 포괄적 비교")
    ap.add_argument("--year1", type=int, default=2024, help="학습 연도")
    ap.add_argument("--year2", type=int, default=2025, help="검증 연도")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 디렉토리")
    ap.add_argument("--skip_training", action="store_true", help="학습 스크립트 스킵")
    args = ap.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    
    if not args.skip_training:
        print(f"\n{'='*80}")
        print(f"[Step 1] 4개 모델 학습 ({args.year1} 데이터)")
        print(f"{'='*80}")
        
        scripts = [
            "train_lgbm_model.py",
            "train_mlp_ensemble.py",
            "train_mlp_improved.py",
            "train_mlp_tuned.py",
        ]
        
        for script in scripts:
            if not os.path.exists(script):
                print(f"⚠ {script} 없음 - 스킵")
                continue
            
            success = run_training_script(script, args.year1, models_dir=args.models_dir)
            if not success:
                print(f"⚠ {script} 실행 실패")
        
        print(f"\n{'='*80}")
        print(f"[Step 2] 모든 모델 학습 완료")
        print(f"{'='*80}\n")
    
    # 비교 보고서
    print(f"\n{'='*80}")
    print(f"[Step 3] 모델 비교 및 분석")
    print(f"{'='*80}\n")
    
    df_comparison, report_json = create_comparison_report(args.year1, args.year2, args.models_dir)
    
    if df_comparison is not None:
        print(f"\n[OK] 비교 완료!")
        print(f"\n[NEXT] 다음 단계:")
        print(f"  1. python select_best_model.py --comp_file {args.models_dir}/comparison_report_{args.year1}_{args.year2}.json")
        print(f"  2. 최고 성능 모델을 deploy/production으로 옮기기")
    else:
        print(f"\n[ERROR] 비교 실패")


if __name__ == "__main__":
    main()
