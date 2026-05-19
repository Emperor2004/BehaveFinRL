import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

app = Flask(__name__, static_folder="static", template_folder="templates")

# Helper function to load joblib logs safely
def load_log_file(filename):
    path = config.LOG_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None

@app.route("/")
def index():
    # Load evaluation logs
    eval_logs = load_log_file("multi_seed_evaluations.joblib")
    
    # Calculate average metrics across seeds for display
    summary_metrics = {}
    seeds_data = []
    
    if eval_logs:
        returns = []
        sharpes = []
        mdds = []
        win_rates = []
        
        bench_ret = 0.0
        bench_sharpe = 0.0
        bench_mdd = 0.0
        
        for seed, data in eval_logs.items():
            m = data["metrics"]
            returns.append(m["cumulative_return"])
            sharpes.append(m["sharpe_ratio"])
            mdds.append(m["max_drawdown"])
            win_rates.append(m["win_rate"])
            
            bench_ret = m["benchmark_return"]
            bench_sharpe = m["benchmark_sharpe"]
            bench_mdd = m["benchmark_mdd"]
            
            seeds_data.append({
                "seed": seed,
                "return": f"{m['cumulative_return']*100:.2f}%",
                "sharpe": f"{m['sharpe_ratio']:.2f}",
                "mdd": f"{m['max_drawdown']*100:.2f}%",
                "win_rate": f"{m['win_rate']*100:.2f}%"
            })
            
        summary_metrics = {
            "avg_return": f"{np.mean(returns)*100:.2f}%",
            "avg_sharpe": f"{np.mean(sharpes):.2f}",
            "avg_mdd": f"{np.mean(mdds)*100:.2f}%",
            "avg_win_rate": f"{np.mean(win_rates)*100:.2f}%",
            "bench_return": f"{bench_ret*100:.2f}%",
            "bench_sharpe": f"{bench_sharpe:.2f}",
            "bench_mdd": f"{bench_mdd*100:.2f}%",
            "outperformance": f"{(np.mean(returns) - bench_ret)*100:+.2f}%"
        }
    else:
        summary_metrics = {
            "avg_return": "N/A", "avg_sharpe": "N/A", "avg_mdd": "N/A", "avg_win_rate": "N/A",
            "bench_return": "N/A", "bench_sharpe": "N/A", "bench_mdd": "N/A", "outperformance": "N/A"
        }

    return render_template("index.html", metrics=summary_metrics, seeds=seeds_data, active_page="overview")

@app.route("/training")
def training():
    return render_template("training.html", active_page="training")

@app.route("/regime")
def regime():
    # Load HMM logical state specs
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    regime_stats = []
    if hmm_path.exists():
        detector = joblib.load(hmm_path)
        # We can extract some summary info about states
        # The transition matrix
        trans_mat = detector.model.transmat_
        # Format transition matrix for display
        trans_data = []
        for i in range(config.HMM_STATES):
            row = []
            for j in range(config.HMM_STATES):
                # Map logical states
                logical_i = detector.state_map.get(i, i)
                logical_j = detector.state_map.get(j, j)
                row.append(f"{trans_mat[i, j]*100:.1f}%")
            trans_data.append(row)
            
        regime_stats = {
            "bull_lambda": config.LAMBDA_BULL,
            "bear_lambda": config.LAMBDA_BEAR,
            "volatile_lambda": config.LAMBDA_VOLATILE,
        }
        
    return render_template("regime.html", stats=regime_stats, active_page="regime")

@app.route("/shap")
def shap_page():
    shap_data = load_log_file("shap_values.joblib")
    has_plots = {
        "bull": (config.BASE_DIR / "dashboard" / "static" / "images" / "shap_importance_bull.png").exists(),
        "bear": (config.BASE_DIR / "dashboard" / "static" / "images" / "shap_importance_bear.png").exists(),
        "volatile": (config.BASE_DIR / "dashboard" / "static" / "images" / "shap_importance_volatile.png").exists(),
    }
    return render_template("shap.html", shap_data=shap_data, has_plots=has_plots, active_page="shap")

# API Endpoints
@app.route("/api/portfolio_data")
def api_portfolio_data():
    """
    Returns timeline of equity curves (portfolio net worth for each seed vs benchmark)
    """
    eval_logs = load_log_file("multi_seed_evaluations.joblib")
    if not eval_logs:
        return jsonify({"error": "No evaluation logs found. Run training first."})
        
    # Get primary timelines from the first seed (all seeds share the same test dates/prices)
    first_seed = list(eval_logs.keys())[0]
    dates = eval_logs[first_seed]["dates"]
    prices = eval_logs[first_seed]["prices"]
    
    # Calculate buy-and-hold benchmark curve (starting at config.INITIAL_BALANCE)
    initial_price = prices[0]
    benchmark_curve = [(p / initial_price) * config.INITIAL_BALANCE for p in prices]
    
    response = {
        "dates": dates,
        "benchmark": benchmark_curve,
        "seeds": {}
    }
    
    for seed, data in eval_logs.items():
        response["seeds"][str(seed)] = data["net_worth_history"]
        
    return jsonify(response)

@app.route("/api/regime_data")
def api_regime_data():
    """
    Returns daily price data merged with HMM regime labels and dynamic lambda.
    """
    test_path = config.DATA_DIR / f"{config.TICKER}_test.csv"
    hmm_path = config.MODEL_DIR / "hmm_regime_detector.joblib"
    
    if not test_path.exists() or not hmm_path.exists():
        return jsonify({"error": "Data or HMM model not found. Complete preprocessing and training."})
        
    df = pd.read_csv(test_path)
    detector = joblib.load(hmm_path)
    
    regimes = detector.predict(df)
    lambdas = detector.get_lambda(regimes)
    
    response = {
        "dates": df["date"].tolist(),
        "prices": df["adj_close"].tolist(),
        "regimes": regimes.tolist(),
        "lambdas": lambdas.tolist(),
        "vix": df["vix"].tolist()
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
