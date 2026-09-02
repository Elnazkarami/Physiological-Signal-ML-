"""Seven defects an external review found, each pinned so it cannot come back.

Every one was real when reported and every one is asserted here against the
behaviour that was wrong, not against the behaviour that is now right -- a test
that only says "flat signal, zero responses" would pass again the day the
padding is changed back, because zero is also what a broken implementation
returns for a signal it happens to handle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from physioml.core.prediction import Prediction
from physioml.io.wesad import MODALITIES, WESADError, modality_of
from physioml.peripheral.chest import check_ecg, check_emg, check_resp
from physioml.peripheral.features import eda_features, extract
from physioml.peripheral.preprocessing import DEFAULT_PREPROCESSING, Preprocessing
from physioml.peripheral.qc import (
    DEFAULT_POLICY,
    MISSING,
    QCResult,
    assess,
    check_acc,
    check_bvp,
    check_eda,
    check_temp,
)
from physioml.peripheral.windowing import epochs
from tests.test_features import epoch_of, pulse

RATE = 4.0


# ── 1. electrodermal decomposition invented events at the boundary ──────────


def test_a_flat_signal_produces_no_skin_conductance_responses():
    """It produced exactly one per minute: the moving average was taken with
    zero padding, so the tonic level dived at each edge and the phasic residual
    spiked there. Every window of every subject carried the artifact."""
    flat = np.full((240, 1), 3.0)
    assert eda_features(flat, RATE)["scr_count_per_min"] == 0.0


@pytest.mark.parametrize("level", [0.5, 3.0, 12.0])
def test_the_artifact_did_not_depend_on_the_level_it_sat_at(level):
    assert eda_features(np.full((240, 1), level), RATE)["scr_count_per_min"] == 0.0


def test_a_gentle_drift_is_not_a_response_either():
    ramp = (3.0 + 0.001 * np.arange(240)).reshape(-1, 1)
    assert eda_features(ramp, RATE)["scr_count_per_min"] == 0.0


def test_responses_that_are_really_there_are_still_counted():
    """The other half of the fix: suppressing the artifact is only correct if
    real events survive."""
    time = np.arange(240) / RATE
    signal = np.full(240, 3.0)
    for onset in (10.0, 25.0, 45.0):
        signal += 0.4 * np.exp(-((time - onset) ** 2) / (2 * 1.5**2)) * (time >= onset)
    found = eda_features(signal.reshape(-1, 1), RATE)
    assert found["scr_count_per_min"] == pytest.approx(3.0)
    assert found["scr_amplitude_mean"] > 0.0


# ── 2. missing data passed quality control ──────────────────────────────────


@pytest.mark.parametrize(
    ("check", "shape", "rate"),
    [
        (check_bvp, (3840, 1), 64.0),
        (check_eda, (240, 1), 4.0),
        (check_temp, (240, 1), 4.0),
        (check_acc, (1920, 3), 32.0),
    ],
)
def test_a_window_of_missing_samples_is_refused(check, shape, rate):
    """Every comparison against NaN is false, so it passed each threshold in
    turn and was marked valid. Its features were then dropped as non-finite and
    the row lost that signal with nothing on record saying why."""
    assert check(np.full(shape, np.nan), rate, DEFAULT_POLICY) == [MISSING]


@pytest.mark.parametrize(
    ("check", "rate"), [(check_ecg, 700.0), (check_resp, 700.0), (check_emg, 700.0)]
)
def test_the_chest_checks_refuse_missing_samples_too(check, rate):
    assert check(np.full(int(rate * 60), np.nan), rate, DEFAULT_POLICY) == [MISSING]


def test_a_single_missing_sample_is_enough_to_refuse():
    """Tolerating "a few" would be a threshold nobody chose."""
    signal = np.full((240, 1), 3.0)
    signal[100] = np.nan
    assert check_eda(signal, RATE, DEFAULT_POLICY) == [MISSING]


# ── 3. an unrecognised signal became accelerometry ──────────────────────────


def test_chest_muscle_activity_is_muscle_activity():
    """It was accelerometry, and the recording then claimed to be something it
    was not for the rest of its life in the provenance chain."""
    assert modality_of("EMG").value == "emg"


def test_an_unknown_signal_is_refused_rather_than_guessed_at():
    with pytest.raises(WESADError, match="unknown signal"):
        modality_of("Gyroscope")


def test_the_error_says_what_the_reader_does_understand():
    with pytest.raises(WESADError, match="ACC"):
        modality_of("Gyroscope")


def test_every_signal_wesad_stores_has_a_modality():
    assert set(MODALITIES) >= {"ACC", "BVP", "ECG", "EDA", "EMG", "Resp", "TEMP", "Temp"}


# ── 6. prediction identity omitted the stated confidence ────────────────────


def prediction(**over):
    base = {
        "study_id": "S",
        "subject_id": "A",
        "task": "stress",
        "window_start": datetime(2020, 1, 1, tzinfo=UTC),
        "window_end": datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=60),
        "predicted_class": "stressed",
        "probability": 0.51,
        "model_name": "m",
        "model_version": "1",
        "training_run_id": "r",
        "feature_set_version": "1",
        "feature_ids": ("f",),
        "source_window_ids": ("w",),
    }
    return Prediction.create(**(base | over))


def test_two_predictions_that_disagree_about_confidence_are_two_predictions():
    """They shared an identity. CDFS writes the class and the confidence as two
    facts derived from one prediction, so a lineage query would have found them
    claiming the same origin while saying different things."""
    assert prediction(probability=0.51).prediction_id != (
        prediction(probability=0.99).prediction_id
    )


def test_the_same_prediction_still_has_the_same_identity():
    """Content addressing is only useful if it is stable."""
    assert prediction().prediction_id == prediction().prediction_id


def test_a_prediction_with_no_stated_confidence_still_has_an_identity():
    assert prediction(probability=None).prediction_id
    assert prediction(probability=None).prediction_id != prediction().prediction_id


# ── 5. preprocessing was recorded but not used ──────────────────────────────


def synthetic_subject(seconds: float = 120.0):
    """One subject of wrist signals, enough to cut windows from."""
    from physioml.core.recording import Modality, Recording
    from physioml.io.wesad import LABEL_HZ, WRIST_HZ, SubjectData

    started = datetime(2017, 1, 1, tzinfo=UTC)
    signals, recordings = {}, {}
    for name, rate in WRIST_HZ.items():
        width = 3 if name == "ACC" else 1
        signals[name] = (
            pulse(seconds=seconds, rate=rate)
            if name == "BVP"
            else np.full((int(seconds * rate), width), 3.0)
        )
        recordings[name] = Recording.create(
            study_id="T",
            subject_id="S00",
            modality=Modality.ACC if name == "ACC" else Modality[name],
            sampling_rate_hz=rate,
            start_time=started,
            duration_seconds=seconds,
            channels=("x", "y", "z") if name == "ACC" else (name,),
        )
    labels = np.full(int(seconds * LABEL_HZ), 2)
    return SubjectData("S00", signals, labels, recordings)


def test_the_window_records_the_settings_the_features_were_computed_under():
    """Windows defaulted to an empty preprocessing identity while the pulse
    extractor filtered with the module default, so the provenance and the
    arithmetic could disagree with nothing to notice."""
    made = epochs(synthetic_subject(), length_seconds=60.0, stride_seconds=60.0)
    assert made
    for window in made[0].windows.values():
        assert window.preprocessing_run_id == DEFAULT_PREPROCESSING.run_id
        assert window.preprocessing_run_id != ""


def test_a_different_filter_is_recorded_on_the_windows_it_produced():
    wider = Preprocessing(version="wider-1.0", bvp_low_hz=0.3, bvp_high_hz=10.0)
    made = epochs(
        synthetic_subject(),
        length_seconds=60.0,
        stride_seconds=60.0,
        preprocessing=wider,
    )
    assert made and made[0].preprocessing is wider
    for window in made[0].windows.values():
        assert window.preprocessing_run_id == wider.run_id


def test_changing_the_filter_changes_what_the_window_claims():
    wider = Preprocessing(version="wider-1.0", bvp_low_hz=0.3, bvp_high_hz=10.0)
    assert wider.run_id != DEFAULT_PREPROCESSING.run_id


def test_the_extractor_filters_with_the_configuration_the_epoch_carries():
    """Not with the module default. A pulse band that excludes the pulse should
    change the answer; if it does not, the setting is not reaching the filter."""
    from dataclasses import replace

    epoch = epoch_of(BVP=pulse(bpm=72.0))
    verdict = assess(epoch, DEFAULT_POLICY)
    normal = extract(epoch, verdict)

    absurd = Preprocessing(version="absurd-1.0", bvp_low_hz=6.0, bvp_high_hz=7.0)
    shifted = replace(epoch, preprocessing=absurd)
    strange = extract(shifted, assess(shifted, DEFAULT_POLICY))

    def rate_of(features):
        found = {f.name: f.value for f in features}
        return found.get("pulse_rate_mean")

    assert rate_of(normal) == pytest.approx(72.0, abs=3.0)
    assert rate_of(strange) != rate_of(normal)


# ── 7. features could be extracted without quality control ──────────────────


def test_extraction_requires_a_verdict():
    """The order the README promises is window, judge, then measure. It was
    skippable by accident, and produced features from signals nothing had
    looked at."""
    epoch = epoch_of(BVP=pulse())
    with pytest.raises(TypeError):
        extract(epoch)  # type: ignore[call-arg]


def test_a_rejected_signal_contributes_nothing_even_when_it_would_compute():
    epoch = epoch_of(BVP=pulse())
    rejected = QCResult(
        statuses={
            "BVP": __import__(
                "physioml.core.window", fromlist=["QCStatus"]
            ).QCStatus.REJECTED
        },
        codes={"BVP": ("no_pulse",)},
        policy_version="test",
    )
    assert extract(epoch, rejected) == []
    assert extract(epoch, assess(epoch, DEFAULT_POLICY)) != []


# ── 4. recordings carried no checksum of their own samples ──────────────────


def test_a_recording_carries_a_checksum_of_its_own_samples():
    """Identity was built from metadata alone -- subject, device, rate,
    duration -- every one of which can stay the same while the samples change.
    A corrected export and the export it corrects were the same recording."""
    from physioml.io.wesad import WESAD
    from tests.paths import WESAD_ARCHIVE as archive
    from tests.paths import WESAD_MISSING

    if not archive.is_file():
        pytest.skip(WESAD_MISSING)

    for name, recording in WESAD(archive).read("S2").recordings.items():
        assert recording.source_hash, f"{name} has no checksum"
        assert len(recording.source_hash) == 64


def test_the_checksum_moves_when_a_single_sample_does():
    import hashlib

    array = np.zeros(1000)
    before = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    array[500] = 1e-12
    after = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    assert before != after


# ── 8. predictions of exactly zero fell into no calibration bin ─────────────


def test_a_model_that_always_says_zero_is_not_perfectly_calibrated():
    """It reported an expected calibration error of exactly 0.000.

    The bins were half-open on both sides -- ``(low, high]`` -- so the first
    was ``(0.0, 0.1]`` and a prediction of exactly 0.0 belonged to no bin at
    all. It was dropped from the weighted average rather than counted. The
    majority-class baseline predicts exactly 0.0 for every row, so the one
    model in every table whose calibration is knowable by hand was the one
    reported wrongly.
    """
    from physioml.evaluation.metrics import expected_calibration_error

    truth = np.array([0] * 78 + [1] * 22)
    assert expected_calibration_error(truth, np.zeros(100)) == pytest.approx(0.22)


def test_it_agrees_with_the_brier_score_for_a_constant_answer():
    """For a model that states one probability, the two must be consistent:
    stating 0.0 on a set that is 22% positive is 0.22 wrong either way."""
    from physioml.evaluation.metrics import expected_calibration_error, score

    truth = np.array([0] * 78 + [1] * 22)
    got = score(truth, np.zeros(100, dtype=int), np.zeros(100))
    assert got.brier == pytest.approx(0.22)
    assert got.ece == pytest.approx(0.22)
    assert expected_calibration_error(truth, np.zeros(100)) == pytest.approx(got.brier)


def test_a_prediction_of_exactly_one_was_always_counted():
    """The bug was asymmetric, which is why it survived: the last bin is
    ``(0.9, 1.0]`` and 1.0 falls inside it."""
    from physioml.evaluation.metrics import expected_calibration_error

    truth = np.array([1] * 78 + [0] * 22)
    assert expected_calibration_error(truth, np.ones(100)) == pytest.approx(0.22)


def test_the_omission_flattered_calibration_specifically():
    """Isotonic regression clips to [0, 1] and reaches the endpoint where an
    uncalibrated model does not, so its predictions were the ones being
    dropped. A calibrator that answers zero must be scored for it."""
    from physioml.evaluation.metrics import expected_calibration_error

    truth = np.array([0] * 90 + [1] * 10)
    calibrated_like = np.concatenate([np.zeros(90), np.zeros(10)])
    assert expected_calibration_error(truth, calibrated_like) == pytest.approx(0.10)


def test_a_constant_at_the_prevalence_is_well_calibrated_and_useless():
    """The reason a calibration number needs a baseline beside it. This
    predictor cannot tell one window from another and minimises the metric."""
    from physioml.evaluation.metrics import expected_calibration_error, score

    truth = np.array([0] * 78 + [1] * 22)
    constant = np.full(100, 0.22)
    assert expected_calibration_error(truth, constant) == pytest.approx(0.0, abs=0.01)
    got = score(truth, np.zeros(100, dtype=int), constant)
    assert got.roc_auc == pytest.approx(0.5), "no discrimination at all"
    assert got.brier > 0.17, "and a proper scoring rule says so"


def test_every_probability_lands_in_exactly_one_bin():
    """The property the fix rests on, over the whole interval."""
    from physioml.evaluation.metrics import expected_calibration_error

    values = np.linspace(0.0, 1.0, 501)
    truth = (values > 0.5).astype(int)
    # If any value were dropped the weights would not sum to one, and a
    # perfectly calibrated-by-construction set would not score zero.
    assert expected_calibration_error(truth, truth.astype(float)) == pytest.approx(0.0)
    assert expected_calibration_error(np.ones(501, dtype=int), values) == pytest.approx(
        float(np.mean(1.0 - values)), abs=0.01
    )
