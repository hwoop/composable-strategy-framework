import numpy as np
import pandas as pd

class NoteDiscretizer:
    """
    Simple time-binning for text notes.

    Args:
        timestep (float): size of each time bin (same units as timestamps).
        start_time (str): 'zero' to align bins from time zero,
                          'relative' to align bins from first note timestamp.
    """
    def __init__(self, timestep=1.0, start_time='zero'):
        self.timestep = float(timestep)
        if start_time not in ('zero', 'relative'):
            raise ValueError("start_time must be 'zero' or 'relative'")
        
        self.start_time = start_time


    def transform(self, df: pd.DataFrame, end=None):
        eps = 1e-6

        ts = df['Hours'].tolist()
        first_time = 0 if self.start_time == 'zero' else ts[0]
        max_hours = (max(ts) - first_time) if end is None else (end - first_time)

        N_bins = int(max_hours / self.timestep + 1.0 - eps)
        binned_notes = ['' for _ in range(N_bins)]

        for _, row in df.iterrows():
            t = row['Hours'] - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self.timestep - eps)
            if 0 <= bin_id < N_bins:
                note = row['TEXT']
                
                if not pd.isna(note):
                    binned_notes[bin_id] += ' ' + note if binned_notes[bin_id] else note

        return np.array(binned_notes)


    def transform_batch(self, dfs: list, end=None):
        """
        여러 환자의 notes를 한번에 binning할 때 사용
        dfs: List of DataFrames
        returns: List of binned notes list per patient
        """
        return [self.transform(df, end) for df in dfs]
    