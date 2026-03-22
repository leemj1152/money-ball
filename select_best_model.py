# select_best_model.py
"""
최고 성능 모델 선택 + 조건부 Git 커밋/푸시:
- 비교 결과 분석
- 기준선(0.5562 AUC) 초과 확인
- 초과 시: 모델 배포 + Git 커밋 + 푸시
- 미달 시: 로그 기록만 하고 스킵

실행:
  python select_best_model.py --comp_file models/comparison_report_2024_2025.json
"""
import argparse
import os
import json
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


BASELINE_AUC = 0.5562  # 기준선
GITHUB_REPO = "https://github.com/leemj1152/money-ball.git"


def load_comparison_report(comp_file):
    """비교 보고서 로드"""
    
    if not os.path.exists(comp_file):
        print(f"❌ 보고서 파일 없음: {comp_file}")
        return None
    
    with open(comp_file, "r") as f:
        report = json.load(f)
    
    return report


def select_best_model(report):
    """최고 성능 모델 선택"""
    
    best_model = report.get("best_model")
    best_auc = report.get("best_auc")
    improvement = report.get("improvement")
    improvement_pct = report.get("improvement_pct")
    
    print(f"\n{'='*80}")
    print(f"최고 성능 모델 선택")
    print(f"{'='*80}")
    print(f"🏆 모델: {best_model}")
    print(f"   AUC: {best_auc:.4f}")
    print(f"   기준선: {BASELINE_AUC:.4f}")
    print(f"   개선: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    return best_model, best_auc, improvement


def check_performance(best_auc, improvement):
    """성능 검사"""
    
    print(f"\n{'='*80}")
    print(f"성능 검사")
    print(f"{'='*80}")
    
    if best_auc > BASELINE_AUC:
        print(f"✅ 기준선 초과 ({best_auc:.4f} > {BASELINE_AUC:.4f})")
        print(f"   → 배포 진행")
        return True
    else:
        print(f"❌ 기준선 미달 ({best_auc:.4f} ≤ {BASELINE_AUC:.4f})")
        print(f"   → 배포 취소")
        return False


def get_model_files(best_model, year, models_dir="models"):
    """모델 파일 목록 반환"""
    
    # 모델 타입별 파일 이름 매핑
    file_mappings = {
        "LightGBM": [
            f"model_lgbm_{year}.txt",
            f"scaler_lgbm_{year}.joblib",
            f"feature_cols_lgbm_{year}.json",
            f"model_meta_lgbm_{year}.json",
            f"feature_importance_lgbm_{year}.json",
        ],
        "MLP Ensemble": [
            f"scaler_ensemble_mlp_{year}.joblib",
            f"feature_cols_ensemble_mlp_{year}.json",
            f"model_meta_ensemble_mlp_{year}.json",
        ] + [f"model_{i}_ensemble_mlp_{year}.pt" for i in range(3)],
        "MLP Improved": [
            f"model_mlp_improved_{year}.pt",
            f"scaler_mlp_improved_{year}.joblib",
            f"feature_cols_mlp_improved_{year}.json",
            f"model_meta_mlp_improved_{year}.json",
        ],
        "MLP Tuned": [
            f"model_mlp_tuned_{year}.pt",
            f"scaler_mlp_tuned_{year}.joblib",
            f"feature_cols_mlp_tuned_{year}.json",
            f"model_meta_mlp_tuned_{year}.json",
        ],
    }
    
    files = file_mappings.get(best_model, [])
    full_paths = [os.path.join(models_dir, f) for f in files]
    
    return full_paths


def deploy_model(best_model, year, models_dir="models", deploy_dir="models/production"):
    """모델 배포 디렉토리로 복사"""
    
    print(f"\n{'='*80}")
    print(f"모델 배포")
    print(f"{'='*80}")
    
    os.makedirs(deploy_dir, exist_ok=True)
    
    model_files = get_model_files(best_model, year, models_dir)
    deployed_files = []
    
    for src in model_files:
        if os.path.exists(src):
            dst = os.path.join(deploy_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            deployed_files.append(os.path.basename(src))
            print(f"✓ {os.path.basename(src)}")
        else:
            print(f"⚠ {os.path.basename(src)} - 없음")
    
    # 배포 메타데이터
    deploy_meta = {
        "model": best_model,
        "year": year,
        "timestamp": datetime.now().isoformat(),
        "files": deployed_files,
        "baseline_auc": BASELINE_AUC,
    }
    
    meta_file = os.path.join(deploy_dir, "deployment_meta.json")
    with open(meta_file, "w") as f:
        json.dump(deploy_meta, f, indent=2)
    
    print(f"\n✅ 배포 완료: {deploy_dir}/")
    
    return deployed_files


def git_commit_and_push(best_model, best_auc, improvement, improvement_pct, year1, year2):
    """Git 커밋 및 푸시"""
    
    print(f"\n{'='*80}")
    print(f"Git 커밋 및 푸시")
    print(f"{'='*80}")
    
    # 커밋 메시지
    commit_msg = f"""🚀 모델 개선: {best_model} → AUC {BASELINE_AUC:.4f} → {best_auc:.4f} (+{improvement_pct:.2f}%)

다중 연도 검증 결과:
- 학습: {year1} 데이터
- 검증: {year2} 데이터
- 모델: {best_model}
- AUC 개선: {improvement:+.4f} ({improvement_pct:+.2f}%)

🎯 성능 메트릭:
- New AUC: {best_auc:.4f}
- Baseline AUC: {BASELINE_AUC:.4f}
- Improvement: {improvement:+.4f}

✅ 모델 배포 완료"""
    
    try:
        # 상태 확인
        print("\n[Git] 상태 확인...")
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode != 0:
            print(f"❌ Git 상태 확인 실패: {result.stderr}")
            return False
        
        # 변경사항 있으면 스테이징
        if result.stdout.strip():
            print("\n[Git] 변경사항 스테이징...")
            subprocess.run(["git", "add", "-A"], cwd=".", check=True)
            print("✓ 모든 변경사항 스테이징 완료")
        
        # 커밋
        print("\n[Git] 커밋...")
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=".",
            check=True
        )
        print("✓ 커밋 완료")
        
        # 푸시
        print("\n[Git] 푸시 (origin/main)...")
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=".",
            check=True
        )
        print("✓ 푸시 완료")
        
        print(f"\n✅ Git 커밋/푸시 성공!")
        print(f"   Remote: {GITHUB_REPO}")
        print(f"   Branch: main")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git 작업 실패: {e}")
        return False


def log_result(report, deployed, approved, year1, year2, log_file="model_selection.log"):
    """결과 로그 기록"""
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.now().isoformat()}] 모델 선택 기록\n")
        f.write(f"{'='*80}\n")
        f.write(f"학습: {year1}, 검증: {year2}\n")
        f.write(f"최고 모델: {report['best_model']}\n")
        f.write(f"AUC: {report['best_auc']:.4f}\n")
        f.write(f"기준선: {BASELINE_AUC:.4f}\n")
        f.write(f"개선: {report['improvement']:+.4f} ({report['improvement_pct']:+.2f}%)\n")
        f.write(f"배포 승인: {approved}\n")
        if deployed:
            f.write(f"배포 완료: 예\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n💾 결과 로그 기록: {log_file}")


def main():
    ap = argparse.ArgumentParser(description="최고 모델 선택 및 Git 푸시")
    ap.add_argument("--comp_file", type=str, required=True, help="비교 보고서 파일 (JSON)")
    ap.add_argument("--year1", type=int, default=2024, help="학습 연도")
    ap.add_argument("--year2", type=int, default=2025, help="검증 연도")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 디렉토리")
    ap.add_argument("--no_push", action="store_true", help="Git 푸시 스킵")
    ap.add_argument("--force", action="store_true", help="기준선 미달 시에도 강제 푸시")
    args = ap.parse_args()
    
    # 보고서 로드
    report = load_comparison_report(args.comp_file)
    if not report:
        sys.exit(1)
    
    # 최고 모델 선택
    best_model, best_auc, improvement = select_best_model(report)
    
    # 성능 검사
    approved = check_performance(best_auc, improvement)
    
    if not approved and not args.force:
        print(f"\n⚠️  배포 취소: 성능 기준 미달")
        log_result(report, False, False, args.year1, args.year2)
        sys.exit(0)
    
    if args.force and not approved:
        print(f"\n⚠️  강제 배포 옵션 사용 (성능 기준 미달 무시)")
    
    # 배포
    deployed_files = deploy_model(best_model, args.year1, args.models_dir)
    
    # Git 푸시
    if not args.no_push:
        git_success = git_commit_and_push(
            best_model, best_auc, improvement,
            report.get("improvement_pct"),
            args.year1, args.year2
        )
    else:
        print(f"\n⊘ Git 푸시 스킵")
        git_success = False
    
    # 로그
    log_result(report, len(deployed_files) > 0, approved, args.year1, args.year2)
    
    print(f"\n{'='*80}")
    print(f"✅ 모델 선택 및 배포 완료")
    print(f"{'='*80}")
    print(f"🏆 배포 모델: {best_model}")
    print(f"📊 AUC: {best_auc:.4f} (기준선 {BASELINE_AUC:.4f})")
    print(f"📈 개선: {improvement:+.4f} ({report.get('improvement_pct'):+.2f}%)")


if __name__ == "__main__":
    main()
