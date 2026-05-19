import numpy as np
import pandas as pd
from validation.cv import PurgedEmbargoKFold

def test_purged_embargo_cv():
    """
    Validates that purging and embargoing windows are mathematically enforced and
    prevent training leakages around the validation boundaries.
    """
    # Create 100 mock daily rows
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=100)})
    
    # 5 splits, purging window = 5 days, embargo window = 5% of dataset (5 days)
    cv = PurgedEmbargoKFold(n_splits=5, purging_window=5, pct_embargo=0.05)
    splits = cv.split(df)
    
    assert len(splits) == 5
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Validation indices must be contiguous
        assert np.all(np.diff(val_idx) == 1)
        
        # Train and validation indices must be completely disjoint
        intersection = np.intersect1d(train_idx, val_idx)
        assert len(intersection) == 0
        
        # Purging check:
        # No train indices can exist in the val_start - purging_window buffer
        val_start = val_idx[0]
        purged_buffer = np.arange(max(0, val_start - 5), val_start)
        assert len(np.intersect1d(train_idx, purged_buffer)) == 0
        
        # Embargoing check:
        # No train indices can exist in the val_end + embargo_window buffer
        val_end = val_idx[-1]
        embargo_window = 5 # max(5, 100 * 0.05)
        embargoed_buffer = np.arange(val_end + 1, min(100, val_end + 1 + embargo_window))
        assert len(np.intersect1d(train_idx, embargoed_buffer)) == 0
