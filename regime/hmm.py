import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

class MarketRegimeDetector:
    """
    Hidden Markov Model (HMM) for market regime detection.
    Fits a GaussianHMM on return and volatility features, sorts states into
    consistent Bull (0), Bear (1), and High Volatility (2) regimes,
    and maps them to adaptive lambda values.
    """
    def __init__(self, n_states=config.HMM_STATES, random_state=config.HMM_RANDOM_STATE):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_map = {}  # HMM raw state -> Ordered logical state (0: Bull, 1: Bear, 2: Volatile)
        self.features_list = ["daily_return", "rolling_volatility", "vix"]

    def _prepare_features(self, df):
        """
        Extracts and scales features for the HMM model.
        """
        for f in self.features_list:
            if f not in df.columns:
                raise ValueError(f"Required feature '{f}' not found in input DataFrame.")
        
        X = df[self.features_list].values
        return X

    def fit(self, df):
        """
        Fits the HMM on the input DataFrame.
        """
        print("Training Hidden Markov Model for market regime detection...")
        X = self._prepare_features(df)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            random_state=self.random_state,
            n_iter=100
        )
        self.model.fit(X_scaled)
        
        # Predict raw states
        raw_states = self.model.predict(X_scaled)
        
        # Resolve HMM label switching: Map raw states to logical states based on returns and volatility
        # We calculate mean return and mean VIX for each raw state
        state_stats = []
        for state in range(self.n_states):
            mask = (raw_states == state)
            state_returns = df.loc[mask, "daily_return"].mean()
            state_vix = df.loc[mask, "vix"].mean()
            state_stats.append({
                "raw_state": state,
                "mean_return": state_returns,
                "mean_vix": state_vix
            })
            
        # 1. Bear (state 1): The state with the lowest mean return (usually highest volatility too)
        state_stats_sorted_return = sorted(state_stats, key=lambda x: x["mean_return"])
        raw_bear = state_stats_sorted_return[0]["raw_state"]
        
        # 2. Bull (state 0): The state with the highest mean return
        raw_bull = state_stats_sorted_return[-1]["raw_state"]
        
        # 3. Volatile (state 2): The intermediate state
        raw_volatile = state_stats_sorted_return[1]["raw_state"]
        
        # Update map: raw state -> logical state
        self.state_map = {
            raw_bull: 0,      # Bull
            raw_bear: 1,      # Bear
            raw_volatile: 2   # High Volatility
        }
        
        print("Logical Regime mapping resolved:")
        print(f"  - Bull Regime (State 0): HMM Raw State {raw_bull} (Mean Return: {self.state_map_info(state_stats, raw_bull)})")
        print(f"  - Bear Regime (State 1): HMM Raw State {raw_bear} (Mean Return: {self.state_map_info(state_stats, raw_bear)})")
        print(f"  - Volatile Regime (State 2): HMM Raw State {raw_volatile} (Mean Return: {self.state_map_info(state_stats, raw_volatile)})")
        
        return self

    def state_map_info(self, stats, raw_state):
        for s in stats:
            if s["raw_state"] == raw_state:
                return f"{s['mean_return']*100:+.4f}%, Mean VIX: {s['mean_vix']:.2f}"
        return "N/A"

    def predict(self, df):
        """
        Predicts logical regime states (0: Bull, 1: Bear, 2: Volatile) for the input DataFrame.
        """
        if self.model is None:
            raise ValueError("HMM Model is not trained yet. Call fit() first.")
            
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        raw_states = self.model.predict(X_scaled)
        
        # Map raw states to logical states
        logical_states = np.array([self.state_map[s] for s in raw_states])
        return logical_states

    def get_lambda(self, regime_state):
        """
        Maps a regime state (scalar or array) to its corresponding Prospect Theory lambda.
        """
        lambda_map = {
            0: config.LAMBDA_BULL,
            1: config.LAMBDA_BEAR,
            2: config.LAMBDA_VOLATILE
        }
        if isinstance(regime_state, np.ndarray):
            return np.vectorize(lambda_map.get)(regime_state)
        return lambda_map.get(regime_state, config.PROSPECT_LAMBDA)

    def save(self, filepath=None):
        """
        Saves the HMM detector instance to disk.
        """
        if filepath is None:
            filepath = config.MODEL_DIR / f"hmm_regime_detector.joblib"
        joblib.dump(self, filepath)
        print(f"HMM model and scaler saved to {filepath}.")

    @classmethod
    def load(cls, filepath=None):
        """
        Loads the HMM detector instance from disk.
        """
        if filepath is None:
            filepath = config.MODEL_DIR / f"hmm_regime_detector.joblib"
        return joblib.load(filepath)

if __name__ == "__main__":
    # Test script if processed data exists
    processed_path = config.DATA_DIR / f"{config.TICKER}_train.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        detector = MarketRegimeDetector()
        detector.fit(df)
        regimes = detector.predict(df)
        df["regime"] = regimes
        df["pt_lambda"] = detector.get_lambda(regimes)
        print(f"HMM test passed. Classified {len(regimes)} days.")
        detector.save()
    else:
        print(f"Processed file {processed_path} not found. Run fetch.py and preprocess.py first.")
