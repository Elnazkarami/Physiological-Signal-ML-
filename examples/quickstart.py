"""The whole pipeline on signals generated here, in one file and no downloads.

    python examples/quickstart.py

WESAD must be requested from its authors and Sleep-EDF Expanded is several
gigabytes, so neither can stand behind an example someone runs to find out
what this library does. The signals below are synthesised instead. Everything
operating on them -- the windowing, the quality control, the feature
extractors, the subject-wise splits, the metrics and the provenance export --
is the same code the measured results in the README came from, and the only
thing replaced is the recording.

THE SCORES THIS PRINTS ARE MEANINGLESS. They describe a difference this script
put into the data on purpose. They are printed because the shape of the output
is the point, and they are labelled at every step so no one lifts them into a
slide. The measured results are in docs/ and in the README.

Requires the signal and ml extras::

    pip install -e ".[signal,ml]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from physioml.core.recording import Recording
from physioml.core.registry import TrainingRun
from physioml.dataset import assemble
from physioml.evaluation.export import (
    as_bundle,
    as_json,
    export_fold,
    parameter_digest,
    unresolved,
)
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import leave_one_subject_out
from physioml.io.wesad import LABEL_HZ, WRIST_HZ, SubjectData, modality_of
from physioml.models.classical import MODELS
from physioml.peripheral.features import extract
from physioml.peripheral.qc import DEFAULT_POLICY, assess
from physioml.peripheral.windowing import epochs

SUBJECTS = [f"P{i:02d}" for i in range(1, 7)]
BASELINE_SECONDS = 210.0
STRESS_SECONDS = 210.0
STUDY = "SYNTHETIC"

# What the generator writes into the signal, which is therefore the most a
# model could possibly recover. Stated here so the printed score can be read
# against the effect that produced it.
EFFECT = """stress blocks are generated with a pulse 9 bpm faster, skin
conductance 0.7 uS higher, skin temperature 0.2 C lower and three times the
movement -- each smaller than the spread this script puts between
participants, so the task is separable but not trivially"""


# ── a recording that does not exist ──────────────────────────────────────────


def wrist_signals(
    rng: np.random.Generator, offsets: dict[str, float]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One synthetic participant: four wrist sensors and a protocol label track.

    Shapes, sampling rates and units match what the Empatica E4 reports in
    WESAD, because the quality control and the extractors are written against
    those and an example that quietly used different ones would be exercising
    nothing.
    """
    blocks = [("baseline", BASELINE_SECONDS, 0.0), ("stress", STRESS_SECONDS, 1.0)]
    signals: dict[str, list[np.ndarray]] = {k: [] for k in WRIST_HZ}
    labels: list[np.ndarray] = []

    for name, seconds, stressed in blocks:
        # Pulse: a fundamental at the heart rate plus a harmonic, which is what
        # gives a photoplethysmogram the in-band concentration quality control
        # looks for. Flat noise would be rejected, correctly.
        bpm = 72.0 + offsets["bpm"] + 9.0 * stressed
        t = np.arange(0.0, seconds, 1.0 / WRIST_HZ["BVP"])
        phase = 2 * np.pi * (bpm / 60.0) * t
        bvp = 60.0 * np.sin(phase) + 18.0 * np.sin(2 * phase + 0.6)
        signals["BVP"].append(bvp + rng.normal(0.0, 9.0, t.size))

        # Skin conductance: a slow tonic level with a few phasic responses on
        # top. Steps stay well under the policy's 1 uS contact-loss limit.
        n = int(seconds * WRIST_HZ["EDA"])
        # Floored at a physiological level: an offset low enough to drive the
        # tonic level onto the clip boundary produced a constant signal, which
        # quality control correctly called a flatline and which then cost the
        # whole cohort nine features.
        tonic = max(1.5, 5.0 + offsets["eda"]) + 0.7 * stressed
        drift = np.cumsum(rng.normal(0.0, 0.02, n))
        eda = tonic + drift
        for onset in rng.integers(0, max(n - 40, 1), 2 + int(4 * stressed)):
            rise = np.exp(-np.arange(40) / 12.0) * rng.uniform(0.15, 0.4)
            eda[onset : onset + 40] += rise[: n - onset]
        signals["EDA"].append(np.clip(eda, 0.05, 59.0))

        temp = (
            33.2 + offsets["temp"] - 0.2 * stressed + np.cumsum(rng.normal(0.0, 0.0015, n))
        )
        signals["TEMP"].append(np.clip(temp, 25.5, 39.5))

        # Accelerometry in E4 device units, where 1 g is 64. Gravity sits on one
        # axis; the stress block moves more, which is what makes the motion flag
        # appear in the quality-control summary below.
        m = int(seconds * WRIST_HZ["ACC"])
        # Movement comes in bouts rather than at a constant level, so the motion
        # flag lands on some of the stress block and not all of it. A constant
        # level put it on every stress window and nowhere else, which made the
        # flag a perfect copy of the label -- a leak the example would then be
        # demonstrating instead of the pipeline.
        bouts = np.repeat(
            rng.gamma(1.6, 1.0, m // int(WRIST_HZ["ACC"] * 10) + 1),
            int(WRIST_HZ["ACC"] * 10),
        )[:m]
        wobble = (0.020 + offsets["acc"]) * (0.6 + 1.1 * stressed) * bouts[:, None]
        acc = rng.normal(0.0, 1.0, (m, 3)) * wobble * DEFAULT_POLICY.acc_g_scale
        acc[:, 2] += DEFAULT_POLICY.acc_g_scale
        signals["ACC"].append(acc)

        code = 1 if name == "baseline" else 2
        labels.append(np.full(int(seconds * LABEL_HZ), code, dtype=np.int16))

    return (
        {k: np.concatenate(v) for k, v in signals.items()},
        np.concatenate(labels),
    )


def digest_of(array: np.ndarray) -> str:
    """A checksum of the samples themselves.

    A recording's identity is otherwise built from metadata alone -- subject,
    device, rate, interval -- every one of which can stay the same while the
    samples change. The WESAD reader computes this; a synthetic recording that
    skipped it would be demonstrating a weaker identity than the real pipeline
    has.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fitted_parameters(pipeline: object) -> list[object]:
    """Every array a fitted pipeline learned, in a fixed order.

    The scaler is part of the fitted state, not packaging: the same
    coefficients applied after a different centring are a different model.
    Taking only the classifier's coefficients would hash half of what was fit.
    """
    found = []
    for _name, step in getattr(pipeline, "steps", [("model", pipeline)]):
        for attribute in ("mean_", "scale_", "var_", "coef_", "intercept_"):
            value = getattr(step, attribute, None)
            if value is not None:
                found.append(value)
    return found


def participant(subject_id: str, seed: int) -> SubjectData:
    """A participant the pipeline cannot tell from a real one."""
    rng = np.random.default_rng(seed)
    offsets = {
        "bpm": rng.normal(0.0, 11.0),
        "eda": rng.normal(0.0, 2.0),
        "temp": rng.normal(0.0, 0.5),
        "acc": abs(rng.normal(0.0, 0.006)),
    }
    signals, labels = wrist_signals(rng, offsets)
    start = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    recordings = {
        name: Recording.create(
            study_id=STUDY,
            subject_id=subject_id,
            modality=modality_of(name),
            sampling_rate_hz=WRIST_HZ[name],
            start_time=start,
            duration_seconds=len(array) / WRIST_HZ[name],
            channels=(
                (f"{name}_x", f"{name}_y", f"{name}_z") if array.ndim == 2 else (name,)
            ),
            device_name="synthetic E4",
            source_uri=f"generated://{subject_id}/{name}",
            source_hash=digest_of(array),
            metadata={"synthetic": "true", "wesad_signal": name},
        )
        for name, array in signals.items()
    }
    return SubjectData(subject_id, signals, labels, recordings)


# ── the pipeline, stage by stage ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline on synthetic signals.")
    parser.add_argument(
        "--write-trace",
        type=Path,
        metavar="PATH",
        help="also write the exported prediction record as JSON. What produces "
        "examples/expected/prediction_trace.json, so the committed file is "
        "output rather than something written by hand to look like output.",
    )
    args = parser.parse_args()

    print(__doc__.split("Requires")[0].strip())
    print("\n" + "=" * 74)

    print("\n[1/5] generating six participants, seven minutes each")
    cohort = [participant(s, seed) for seed, s in enumerate(SUBJECTS)]
    print(
        f"      {len(cohort)} participants, "
        f"{sum(len(c.labels) for c in cohort) / LABEL_HZ / 60:.0f} minutes of signal"
    )

    print("\n[2/5] windowing, quality control, feature extraction")
    rows: list[dict] = []
    meta: list[tuple[str, str, str, float, tuple[str, ...], tuple[str, ...], float]] = []
    codes: dict[str, int] = {}
    unlabelled = 0
    # Kept so the exported chain can be published with its references resolved
    # rather than merely named. The table stores identifiers, not objects, and
    # the builder is the only thing that still holds the objects.
    every_window: dict[str, object] = {}
    every_recording: dict[str, object] = {}

    for data in cohort:
        every_recording.update({r.recording_id: r for r in data.recordings.values()})
        for epoch in epochs(data, length_seconds=60.0, stride_seconds=5.0):
            # A window straddling two conditions has no single label, so it is
            # not a training example. The real builder drops these too.
            if not epoch.labelled:
                unlabelled += 1
                continue

            # The order the README promises: window, judge, then measure.
            # extract() requires the verdict rather than accepting its absence,
            # so features cannot come out of signal nothing has looked at.
            verdict = assess(epoch, DEFAULT_POLICY)
            for signal, reasons in verdict.codes.items():
                for code in reasons:
                    key = f"{signal}:{code}"
                    codes[key] = codes.get(key, 0) + 1

            features = extract(epoch, verdict, DEFAULT_POLICY)
            if not features:
                continue
            rows.append({f.qualified_name: f for f in features})
            every_window.update({w.window_id: w for w in epoch.windows.values()})
            meta.append(
                (
                    data.subject_id,
                    epoch.label or "",
                    next(iter(epoch.windows.values())).window_id,
                    epoch.start_seconds,
                    tuple(sorted(w.window_id for w in epoch.windows.values())),
                    tuple(sorted(w.recording_id for w in epoch.windows.values())),
                    next(iter(epoch.windows.values())).start_time.timestamp(),
                )
            )

    print(
        f"      {len(rows)} labelled windows, {unlabelled} dropped for "
        f"spanning two conditions"
    )
    print(f"      quality control flagged: {codes or 'nothing'}")

    print("\n[3/5] assembling the table")
    table = assemble(
        rows,
        subjects=[m[0] for m in meta],
        labels=[m[1] for m in meta],
        window_ids=[m[2] for m in meta],
        starts=[m[3] for m in meta],
        windows=[m[4] for m in meta],
        recordings=[m[5] for m in meta],
        feature_set_version="synthetic-wrist-1",
        qc_policy_version=DEFAULT_POLICY.version,
        qc_codes=codes,
        start_times=[m[6] for m in meta],
        window_seconds=60.0,
    )
    print(f"      {table.summary()}")
    print(
        f"      every row names its feature vector, "
        f"{len(table.row_windows[0])} signal windows and "
        f"{len(table.row_recordings[0])} recordings"
    )

    print("\n[4/5] leave-one-participant-out evaluation")
    print(
        textwrap.fill(
            f"NOTE: synthetic data. {' '.join(EFFECT.split())}. These numbers "
            "describe the generator, not a result.",
            width=70,
            initial_indent="      ",
            subsequent_indent="      ",
        )
    )
    run = evaluate(
        table,
        MODELS["logistic"],
        leave_one_subject_out(table.subject_ids),
        model_name="logistic",
        task="stress",
        positive="stress",
        dataset_version="synthetic-1",
    )
    s = run.summary
    print(
        f"      balanced accuracy {s['balanced_accuracy_mean']:.3f} "
        f"±{s['balanced_accuracy_sd']:.3f}  "
        f"AUC {s.get('roc_auc_mean', float('nan')):.3f}  "
        f"worst participant {s['worst_subject_balanced_accuracy']:.3f}"
    )
    print("      no participant appears on both sides of any split")

    # Worth printing when it happens, because it is the finding that most often
    # gets read as a broken model. A participant can be ranked perfectly and
    # still score at chance, if the cohort's decision threshold sits off to one
    # side of their whole distribution. docs/wesad-stress.md has the real case.
    worst = min(run.folds, key=lambda f: f.balanced_accuracy)
    if worst.balanced_accuracy < 0.55 and (worst.roc_auc or 0.0) > 0.9:
        who = next(iter(worst.per_subject))
        print(
            textwrap.fill(
                f"note: {who} scores {worst.balanced_accuracy:.3f} at AUC "
                f"{worst.roc_auc:.3f} -- every stressed window ranked above every "
                "calm one, and all of them labelled the same way. A threshold, not "
                "a failure to learn.",
                width=70,
                initial_indent="      ",
                subsequent_indent="      ",
            )
        )

    print("\n[5/5] exporting one fold as traceable predictions")
    split = next(iter(leave_one_subject_out(table.subject_ids)))
    train, test = split.mask(table.subjects)
    model = MODELS["logistic"]()
    model.fit(table.values[train], table.binary("stress")[train])
    predicted = model.predict(table.values[test])
    probability = model.predict_proba(table.values[test])[:, 1]
    scored = test.tolist()

    fold = TrainingRun.create(
        task="stress",
        dataset_version="synthetic-1",
        split_strategy=split.strategy,
        train_subjects=split.train_subjects,
        test_subjects=split.test_subjects,
        feature_schema_version=table.feature_set_version,
        preprocessing_version=table.qc_policy_version,
        random_seed=split.seed,
    )
    artifact, predictions = export_fold(
        table,
        fold,
        scored,
        predicted,
        model_name="logistic",
        model_version="1.0",
        probability=probability,
        study_id=STUDY,
        # A digest of the coefficients that came out, so the artifact's
        # identity covers the fitted object and not only the request that
        # produced it. Without it two models fitted by the same run on the
        # same columns share an identifier however different their parameters.
        artifact_hash=parameter_digest(*fitted_parameters(model)),
    )
    print(
        f"      held out {split.test_subjects[0]}, {len(predictions)} predictions exported"
    )

    one = predictions[0]
    row = test[0]
    print("\n--- the chain behind one prediction " + "-" * 38)
    print(f"prediction      {one.prediction_id}")
    print(f"  participant   {one.subject_id}")
    print(
        f"  window        {one.window_start:%Y-%m-%d %H:%M:%S} to "
        f"{one.window_end:%H:%M:%S} UTC"
    )
    print(
        f"                {table.window_starts[row]:.0f}s into a recording "
        f"that began at "
        f"{cohort[0].recordings['BVP'].start_time:%Y-%m-%d %H:%M} UTC"
    )
    says = "stress" if one.predicted_class == "1" else "not stress"
    print(f"  says          {one.predicted_class!r} ({says}) at p={one.probability:.3f}")
    print(f"  from vector   {one.feature_ids[0]}")
    print(
        f"    computed over {len(table.feature_names)} features, "
        f"set {one.feature_set_version}"
    )
    print(f"  over windows  {', '.join(one.source_window_ids)}")
    print(f"    of recordings {', '.join(table.row_recordings[row])}")
    print(f"  by artifact   {artifact.model_id}")
    print(
        f"    model       {artifact.model_name} {artifact.model_version}, "
        f"fitted by run {artifact.training_run_id}"
    )
    print(f"    parameters  sha256:{artifact.artifact_hash[:32]}...")
    print(
        f"    trained on  {len(split.train_subjects)} participants, "
        f"holding out {split.test_subjects[0]}"
    )
    print(f"  source facts  {one.source_fact_ids or 'none -- not sourced through CDFS'}")
    print("-" * 74)

    print(
        textwrap.fill(
            "Every identifier above is a content hash, and each covers a "
            "different thing. A recording identifier covers its metadata and a "
            "sha256 of the samples. A window covers the recording, the sample "
            "bounds and the preprocessing. A feature vector covers the feature "
            "values. An artifact covers the model, the run, the expected "
            "columns and a digest of the fitted coefficients. A prediction "
            "covers all of that plus the answer and its probability.",
            width=74,
        )
    )
    print(
        textwrap.fill(
            "So recomputing any of these from the same inputs reproduces the "
            "identifier, and changing an input it covers cannot. What they do "
            "not do is certify that the samples on disk are unchanged since "
            "they were read -- that is what the recording's source_hash is "
            "for, and it is only as good as the moment it was taken. Nothing "
            "here is a database key, and no external service was involved.",
            width=74,
        )
    )

    record = as_json(artifact, predictions[:1])
    print("\nas_json(), the form written beside a score (references, not records):")
    print(json.dumps(record["predictions"][0], indent=2))

    if args.write_trace:
        args.write_trace.parent.mkdir(parents=True, exist_ok=True)
        # The bundle rather than as_json: every reference resolved, so the
        # published file stands on its own instead of naming records that live
        # somewhere the reader has no access to.
        referenced = {w for p_ in predictions for w in p_.source_window_ids}
        full = as_bundle(
            artifact,
            predictions,
            table=table,
            rows=scored,
            run=fold,
            windows=[every_window[w] for w in sorted(referenced)],
            recordings=[
                every_recording[r]
                for r in sorted({r for row_ in scored for r in table.row_recordings[row_]})
            ],
        )
        dangling = unresolved(full)
        if dangling:
            raise SystemExit(
                f"the exported chain refers to {len(dangling)} records it does "
                f"not contain: {dangling[:3]}"
            )
        args.write_trace.write_text(json.dumps(full, indent=2) + "\n")
        # To stderr, so that stdout is the same whether or not the flag is
        # passed and one command can regenerate both committed files.
        print(
            f"wrote {len(full['predictions'])} predictions, "
            f"{len(full['records']['feature_vectors'])} feature vectors, "
            f"{len(full['records']['windows'])} windows and "
            f"{len(full['records']['recordings'])} recordings to "
            f"{args.write_trace}; every reference resolves",
            file=sys.stderr,
        )

    print("\n" + "=" * 74)
    print("Synthetic data. The scores above are a property of this script.")
    print("Measured results: docs/wesad-stress.md and docs/sleep.md")


if __name__ == "__main__":
    main()
