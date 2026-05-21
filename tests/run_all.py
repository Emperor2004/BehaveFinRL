import sys
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import test functions
from tests.test_reward import (
    test_reward_neutral,
    test_reward_gains,
    test_reward_losses,
    test_regime_adaptive_losses
)
from tests.test_hmm import (
    test_hmm_loading,
    test_hmm_prediction,
    test_hmm_lambda_mapping
)
from tests.test_env import (
    test_env_initialization,
    test_env_continuous_weight_transitions,
    test_env_prospect_theory_reward
)
from tests.test_cv import (
    test_purged_embargo_cv
)

def run_test(test_func):
    name = test_func.__name__
    print(f"Running {name:.<45} ", end="")
    try:
        test_func()
        print("[  PASS  ]")
        return True
    except AssertionError as e:
        print("[  FAIL  ]")
        print(f"Assertion Error in {name}: {e}")
        return False
    except Exception as e:
        print("[  ERROR ]")
        print(f"Unexpected error in {name}: {e}")
        traceback.print_exc()
        return False

def main():
    print("==================================================")
    print("           BehaveFinRL Test Suite Runner          ")
    print("==================================================")

    test_cases = [
        # Reward function tests
        test_reward_neutral,
        test_reward_gains,
        test_reward_losses,
        test_regime_adaptive_losses,
        
        # HMM model tests
        test_hmm_loading,
        test_hmm_prediction,
        test_hmm_lambda_mapping,
        
        # Gym trading env tests
        test_env_initialization,
        test_env_continuous_weight_transitions,
        test_env_prospect_theory_reward,
        
        # Data partitioning tests
        test_purged_embargo_cv
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        if run_test(test):
            passed += 1
            
    print("==================================================")
    print(f"Test Summary: {passed}/{total} passed")
    print("==================================================")
    
    if passed == total:
        print("All tests completed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed or encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
