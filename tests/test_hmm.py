import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from regime.hmm import MarketRegimeDetector

def test_hmm_loading():
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    assert hmm_path.exists(), "HMM model file must exist (run training first)"
    
    detector = MarketRegimeDetector.load(hmm_path)
    assert detector.model is not None
    assert len(detector.state_map) == config.HMM_STATES

def test_hmm_prediction():
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    detector = MarketRegimeDetector.load(hmm_path)
    
    # Create dummy data containing the required features
    dummy_df = pd.DataFrame({
        "daily_return": [0.001, -0.015, 0.000],
        "rolling_volatility": [0.010, 0.030, 0.015],
        "vix": [15.0, 45.0, 20.0]
    })
    
    regimes = detector.predict(dummy_df)
    assert len(regimes) == 3
    # Check regime values are within 0, 1, 2
    for r in regimes:
        assert r in [0, 1, 2]

def test_hmm_lambda_mapping():
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    detector = MarketRegimeDetector.load(hmm_path)
    
    assert detector.get_lambda(0) == config.LAMBDA_BULL
    assert detector.get_lambda(1) == config.LAMBDA_BEAR
    assert detector.get_lambda(2) == config.LAMBDA_VOLATILE
    
    # Array mapping
    arr = np.array([0, 1, 2])
    lambdas = detector.get_lambda(arr)
    np.testing.assert_array_equal(lambdas, [config.LAMBDA_BULL, config.LAMBDA_BEAR, config.LAMBDA_VOLATILE])
