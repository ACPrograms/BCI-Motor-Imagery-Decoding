import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Upgraded classifier
from sklearn.model_selection import ShuffleSplit, cross_val_score
import warnings
import gc

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Initializing Advanced CSP-LDA Pipeline...")
subjects = range(1, 21)
runs = [4, 8, 12]

subject_scores = []

for subject in subjects:
    try:
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)

        mne.datasets.eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        
        raw.resample(80., npad='auto', verbose=False)

        # OPTIMIZATION 1: Tightened frequency band to isolate the Mu-rhythm (8-15 Hz)
        raw.filter(8., 15., fir_design='firwin', skip_by_annotation='edge', verbose=False)

        events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

        # OPTIMIZATION 2: Narrowed the epoch window. Motor imagery fatigues quickly. 
        # We only look at the first 2 seconds after the cue, cutting out later noise.
        epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=2.0, proj=True, picks=picks,
                            baseline=None, preload=True, verbose=False)

        X = epochs.get_data(copy=False)
        y = epochs.events[:, -1]

        # OPTIMIZATION 3: Swapped SVM for Linear Discriminant Analysis (LDA)
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP', csp), ('LDA', lda)])

        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=2)

        mean_score = np.mean(scores)
        subject_scores.append(mean_score)
        print(f"Subject {subject} calibrated. Accuracy: {mean_score * 100:.2f}%")

        del raw, raws, epochs, X, y
        gc.collect()

    except Exception as e:
        print(f"Skipping Subject {subject} due to error: {e}")

print("\n=== FINAL ADVANCED INTRA-SUBJECT RESULTS ===")
print(f"Average Model Accuracy across {len(subject_scores)} subjects: {np.mean(subject_scores) * 100:.2f}%")
print("============================================")