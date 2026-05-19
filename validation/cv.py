import numpy as np
import pandas as pd

class PurgedEmbargoKFold:
    """
    Marcos López de Prado's Purged and Embargoed Cross-Validation.
    Prevents data leakage and lookahead bias in overlapping financial time series.
    
    For each split:
    - Validation Fold: [val_start, val_end]
    - Purged Range (before): [val_start - purging_window, val_start - 1]
    - Embargoed Range (after): [val_end + 1, val_end + embargo_window]
    - Training Fold: All indices outside validation, purged, and embargoed ranges.
    """
    def __init__(self, n_splits=5, purging_window=20, pct_embargo=0.01):
        self.n_splits = n_splits
        self.purging_window = purging_window
        self.pct_embargo = pct_embargo

    def split(self, df):
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        # Determine base fold size
        fold_size = n_samples // self.n_splits
        embargo_window = int(np.ceil(n_samples * self.pct_embargo))
        # Ensure embargo window is at least equal to purging window for safety
        embargo_window = max(embargo_window, self.purging_window)
        
        splits = []
        for i in range(self.n_splits):
            # Define validation boundaries
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Validation indices
            val_indices = indices[val_start:val_end]
            
            # Determine training exclusion boundaries
            train_exclude_start = max(0, val_start - self.purging_window)
            train_exclude_end = min(n_samples, val_end + embargo_window)
            
            # Train indices
            train_indices = np.setdiff1d(
                indices, 
                indices[train_exclude_start:train_exclude_end]
            )
            
            splits.append((train_indices, val_indices))
            
            print(f"Fold {i+1}/{self.n_splits}:")
            print(f"  - Train samples: {len(train_indices)} (indices outside [{train_exclude_start}, {train_exclude_end-1}])")
            print(f"  - Val samples:   {len(val_indices)} (indices [{val_start}, {val_end-1}])")
            
        return splits
