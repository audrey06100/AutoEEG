import mne
import numpy as np
import os
from glob import glob
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode
from mindware.components.utils.constants import CLASSIFICATION

class EEGDataLoader:
    """
    A utility helper for simple directory structures.
    For complex datasets (BIDS, multiple runs per sub in different folders),
    users are encouraged to manually construct the EEGDataNode.
    """
    def __init__(self, paths, event_id=None):
        if isinstance(paths, str) and os.path.isdir(paths):
            self.file_paths = sorted(glob(os.path.join(paths, "**", "*.set"), recursive=True) + \
                                     glob(os.path.join(paths, "**", "*.edf"), recursive=True) + \
                                     glob(os.path.join(paths, "**", "*.vhdr"), recursive=True))
        elif isinstance(paths, list):
            self.file_paths = paths
        else:
            self.file_paths = [paths]
            
        self.event_id = event_id

    def load_data(self):
        raw_list = []
        event_list = []
        subject_ids = []

        for path in self.file_paths:
            try:
                # 1. Load Raw
                if path.endswith('.set'):
                    raw = mne.io.read_raw_eeglab(path, preload=False)
                elif path.endswith('.edf'):
                    raw = mne.io.read_raw_edf(path, preload=False)
                elif path.endswith('.vhdr'):
                    raw = mne.io.read_raw_brainvision(path, preload=False)
                else: continue
                
                # 2. Find Events
                try:
                    events = mne.find_events(raw, verbose=False)
                except:
                    events, _ = mne.events_from_annotations(raw, verbose=False)
                
                if len(events) == 0: continue

                raw_list.append(raw)
                event_list.append(events)
                
                # Default subject_id: parent folder name or filename prefix
                # User can override this logic as needed
                sub_id = os.path.basename(os.path.dirname(path)) or os.path.basename(path).split('_')[0]
                subject_ids.append(sub_id)
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        if not raw_list:
            raise ValueError("No valid EEG data found in the specified paths.")

        return EEGDataNode(data=[raw_list, event_list], 
                           feature_type=['numeric'] * len(raw_list[0].ch_names),
                           task_type=CLASSIFICATION,
                           subject_ids=subject_ids,
                           info=raw_list[0].info)
