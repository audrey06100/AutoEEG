from mindware.components.feature_engineering.task_space import set_stage
from autoeeg.components.feature_engineering.transformations import (
    Resampler, BandpassFilter, ReReferencer, Segmenter, CSPTransformer, EmptyTransformer
)
from autoeeg.components.models.classification import LDA
from autoeeg.estimators import EEGClassifier
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode
from mindware.components.utils.constants import CLASSIFICATION

# 1. Define Search Space
fe_candidates = {
    'resampler': [Resampler, EmptyTransformer],
    'filter': [BandpassFilter, EmptyTransformer],
    're-referencer': [ReReferencer, EmptyTransformer],
    'segmenter': [Segmenter],
    'spatial_filter': [CSPTransformer, EmptyTransformer]
}
algo_candidates = {'lda': LDA}

set_stage(list(fe_candidates.keys()), fe_candidates)

# 2. Data Loading Options
# Option A: Use Simple Helper (for flat folders)
# from autoeeg.datasets.eeg_dataset import EEGDataLoader
# data_loader = EEGDataLoader(paths="data/simple_folder")
# eeg_data_node = data_loader.load_data()

# Option B: Manual Construction (Recommended for complex datasets)
# This allows handling multiple runs per subject, BIDS, etc.
import mne
raw_list = [mne.io.read_raw_edf("sub01_run01.edf"), mne.io.read_raw_edf("sub01_run02.edf")]
event_list = [mne.find_events(r) for r in raw_list]
subject_ids = ["sub01", "sub01"] # Both belong to sub01

eeg_data_node = EEGDataNode(
    data=[raw_list, event_list],
    feature_type=['numeric'] * len(raw_list[0].ch_names),
    task_type=CLASSIFICATION,
    subject_ids=subject_ids
)

# 3. Initialize and Run
clf = EEGClassifier(
    fe_candidates=fe_candidates,
    algo_candidates=algo_candidates,
    time_limit=3600,
    evaluation_strategy='cv',
    resampling_params={
        'method': 'kfold',      # 'kfold' or 'loso'
        'split_by': 'subject',   # 'subject' or 'trial'
        'folds': 5
    },
    tree_id=0
)

# clf.fit(eeg_data_node)
print("AutoEEG: Flexible Data Entry and Scientific Matrix Evaluation ready.")
