import argparse
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from data.fetch import fetch_stock_data, fetch_macro_indicators
from data.preprocess import preprocess_data
from regime.hmm import MarketRegimeDetector
from models.agent import run_multi_seed_training

def main():
    parser = argparse.ArgumentParser(description="BehaveFinRL Unified Training Pipeline")
    parser.add_argument("--force-fetch", action="store_true", help="Force fetching data from APIs even if raw files exist")
    parser.add_argument("--force-train-hmm", action="store_true", help="Force training HMM detector even if saved model exists")
    parser.add_argument("--quick", action="store_true", help="Run quick RL training (2048 steps per seed) for compilation/test check")
    args = parser.parse_args()

    print("==================================================")
    print("      BehaveFinRL Unified Training Pipeline        ")
    print("==================================================")

    # Step 1: Check/Fetch Raw Data
    stock_raw = config.DATA_DIR / f"{config.TICKER}_raw.csv"
    macro_raw = config.DATA_DIR / "macro_raw.csv"
    
    if args.force_fetch or not stock_raw.exists() or not macro_raw.exists():
        print("\n--- Phase 1: Fetching Data ---")
        fetch_stock_data()
        fetch_macro_indicators()
    else:
        print("\n--- Phase 1: Raw data found, skipping fetch ---")
        
    # Step 2: Preprocess Data
    print("\n--- Phase 1: Preprocessing Data ---")
    df_features, train_df, test_df = preprocess_data()
    
    # Step 3: Check/Train HMM Detector
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    if args.force_train_hmm or not hmm_path.exists():
        print("\n--- Phase 2: Training HMM Detector ---")
        detector = MarketRegimeDetector()
        detector.fit(train_df)
        detector.save()
    else:
        print("\n--- Phase 2: Trained HMM Detector found, loading ---")
        
    # Step 4: Run Multi-Seed RL Agent Training & Evaluation
    print("\n--- Phase 3: Running Reinforcement Learning Training & Evaluation ---")
    run_multi_seed_training(quick_run=args.quick)
    
    print("\n==================================================")
    print("      Pipeline Execution Finished Successfully     ")
    print("==================================================")

if __name__ == "__main__":
    main()
