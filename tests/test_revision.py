import contextlib, io, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from scipy.signal import get_window
from utils.preprocessing import PHICalculator, STFTSoftLabelGenerator
from utils.condition_gate import STFTRateGate
from utils.revision_data import read_manifest, windows
from utils.metrics import calculate_nasa_score
from utils.dataset import RULDataset
from utils.data_loader import DataLoaderFileBase
from models.lstm import LSTMModel
from models.condition_transformer import ConditionTransformer
from train_revision import predict
from utils.csv_io import read_csv as read_sensor_csv

torch.set_num_threads(2)


def trajectory(n=900, slope=0.005, offset=75.0):
    c = np.arange(n) * 100.0
    r = offset + slope * np.arange(n) + 0.01 * np.sin(np.arange(n) / 30)
    return pd.DataFrame(
        dict(cumulative_contact=c, res=r, force=np.full(n, 26.0), RUL=c[-1] - c)
    )


class RevisionContracts(unittest.TestCase):
    def test_csv_utf8_bom_and_cp949(self):
        with tempfile.TemporaryDirectory() as p:
            d = trajectory()
            d["메모"] = "측정"
            for encoding in ["utf-8-sig", "cp949"]:
                file = Path(p) / (encoding + ".csv")
                d.to_csv(file, index=False, encoding=encoding)
                got = read_sensor_csv(file)
                self.assertIn("cumulative_contact", got)
                np.testing.assert_allclose(got.res, d.res)

    def test_phi_does_not_use_future_minimum(self):
        r = np.r_[np.linspace(75, 80, 300), np.full(300, 1.0)]
        f = np.full(600, 26.0)
        m = PHICalculator()
        np.testing.assert_allclose(
            m.calculate_phi(r, f)[0][:300], m.calculate_phi(r[:300], f[:300])[0]
        )
        self.assertFalse(
            np.allclose(
                PHICalculator(baseline_method="legacy_full_min").calculate_phi(r, f)[0][
                    :300
                ],
                m.calculate_phi(r, f)[0][:300],
            )
        )

    def test_dc_feature_equals_weighted_time_means(self):
        d = trajectory()
        r = d.res.to_numpy()
        c = d.cumulative_contact.to_numpy()
        gate = STFTRateGate()
        actual = gate.features(r, c)
        baseline = r[:100].mean()
        frames = np.lib.stride_tricks.sliding_window_view(r - baseline, 256)[::128]
        w = get_window("hann", 256)
        dc = abs((frames * w).sum(1) / w.sum())
        t = (np.arange(len(dc)) * 128 + 128) * 100 / 100000.0
        t -= t.mean()
        expected = np.r_[
            np.array([dc.mean(), dc.std(), *np.percentile(dc, [25, 50, 75])])
            / (c[-1] / 100000.0),
            dc @ t / (t @ t),
            baseline,
            c[-1] / 380000.0,
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_gate_rejects_irregular_cycle_units(self):
        d = trajectory()
        d.loc[30, "cumulative_contact"] += 1
        with self.assertRaises(ValueError):
            STFTRateGate().features(d.res, d.cumulative_contact)

    def test_gate_group_selection_and_save_reload(self):
        x = []
        y = []
        groups = []
        for group in range(4):
            for cls in range(3):
                d = trajectory(slope=0.003 + cls * 0.003, offset=75 + group * 0.1)
                x.append(STFTRateGate().features(d.res, d.cumulative_contact))
                y.append(cls)
                groups.append(group)
        gate = STFTRateGate().fit(x, y, groups)
        self.assertEqual(gate.selected["selection"], "train_group_cv")
        self.assertEqual(len(gate.cv_results), 7)
        with tempfile.TemporaryDirectory() as p:
            path = Path(p) / "gate.pkl"
            gate.save(path)
            reloaded = STFTRateGate.load(path)
            d = trajectory()
            np.testing.assert_allclose(
                gate.predict_weights(d.res, d.cumulative_contact),
                reloaded.predict_weights(d.res, d.cumulative_contact),
            )

    def test_manifest_requires_explicit_rul(self):
        with tempfile.TemporaryDirectory() as p:
            p = Path(p)
            trajectory().drop(columns="RUL").to_csv(p / "a.csv", index=False)
            pd.DataFrame(
                [
                    dict(
                        record_id="a",
                        path="a.csv",
                        split="train",
                        condition="0A",
                        group_id="a",
                    )
                ]
            ).to_csv(p / "manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "never silently"):
                read_manifest(p / "manifest.csv")

    def test_manifest_overlap_is_explicit(self):
        with tempfile.TemporaryDirectory() as p:
            p = Path(p)
            trajectory().to_csv(p / "a.csv", index=False)
            rows = [
                dict(
                    record_id=str(i),
                    path="a.csv",
                    split="train",
                    condition=c,
                    group_id="shared",
                )
                for i, c in enumerate(["0A", "0.5A", "1.0A"])
            ]
            rows.append(
                dict(
                    record_id="v",
                    path="a.csv",
                    split="val",
                    condition="0.7A",
                    group_id="shared",
                )
            )
            pd.DataFrame(rows).to_csv(p / "m.csv", index=False)
            with self.assertRaisesRegex(ValueError, "overlap"):
                read_manifest(p / "m.csv")
            _, overlap = read_manifest(p / "m.csv", allow_overlap=True)
            self.assertEqual(overlap, ["shared"])

    def test_measured_cycle_is_independent_of_rul(self):
        a = trajectory().to_numpy()
        a[:, 3] = 12345.0
        r = dict(array=a, condition="0A", record_id="a", group_id="a")
        x, _, _, meta = windows(
            [r], dict(mean=np.zeros(4), scale=np.ones(4)), training=False
        )
        self.assertAlmostEqual(
            x[0, -1, -1], meta[0]["observed_cycle"] / 380000.0, places=6
        )
        old = RULDataset(
            [
                dict(
                    inputs=np.zeros((900, 4)),
                    rul=a[:, 3],
                    rul_original=a[:, 3],
                    cumulative_contact=a[:, 0],
                )
            ],
            window_size=200,
            stride=100,
        )
        self.assertGreater(float(old[0][0][0, -1]), 0.0)

    def test_nasa_perfect_prediction_and_direction(self):
        target = np.array([100000.0])
        self.assertEqual(calculate_nasa_score(target, target), 0.0)
        self.assertGreater(
            calculate_nasa_score(target + 10000, target),
            calculate_nasa_score(target - 10000, target),
        )

    def test_legacy_split_has_no_test_file_substitution(self):
        import random

        with tempfile.TemporaryDirectory() as p:
            folder = Path(p) / "0A"
            folder.mkdir()
            for i in range(5):
                (folder / f"{i}.csv").write_text("x", encoding="utf-8")
            cfg = SimpleNamespace(
                ALPHA=2.34,
                BETA=0.133,
                SOFT_LABEL_METHOD="RF",
                MAX_RUL_CYCLE=380000.0,
                MIN_RUL_CYCLE=0.0,
                DATA_ROOT=p,
                FOLDERS={"0A": "0A"},
                N_TRAIN=1,
                N_VAL=1,
                N_TEST=1,
                SEED=42,
            )
            loader = DataLoaderFileBase(cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                loader.load_and_split_files()
            paths = sorted(str(v) for v in folder.glob("*.csv"))
            random.Random(42).shuffle(paths)
            self.assertEqual(loader.test_files["0A"], paths[2:3])

    def test_stft_scaler_fits_training_files_only(self):
        with tempfile.TemporaryDirectory() as p:
            p = Path(p)
            train = {}
            val = {}
            expected = []
            gate = STFTSoftLabelGenerator()
            gate.device = torch.device("cpu")
            for j, c in enumerate(["0A", "0.5A", "1.0A"]):
                a = trajectory(n=600, offset=70 + j)
                b = trajectory(n=600, offset=170 + j)
                t = p / f"t{j}.csv"
                v = p / f"v{j}.csv"
                a.to_csv(t, index=False)
                b.to_csv(v, index=False)
                train[c] = [t]
                val[c] = [v]
                for start in range(0, 201, 100):
                    expected.append(
                        gate.extract_hybrid_features_window(
                            a.res.to_numpy()[start : start + 400]
                        )
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                gate.train_mlp(
                    train,
                    validation_data_dict=val,
                    window_size=400,
                    stride=100,
                    epochs=2,
                    batch_size=4,
                )
            np.testing.assert_allclose(
                gate.scaler.mean_, np.mean(expected, axis=0), rtol=1e-10, atol=1e-10
            )
            self.assertEqual(gate.validation_protocol["scaler_fit"], "training_only")
            self.assertFalse(
                set(gate.validation_protocol["train_groups"])
                & set(gate.validation_protocol["validation_groups"])
            )

    def test_lstm_restores_best_epoch_in_memory(self):
        with tempfile.TemporaryDirectory() as p:
            cfg = SimpleNamespace(
                get_device=lambda: torch.device("cpu"),
                LSTM_HIDDEN_SIZE=4,
                LSTM_NUM_LAYERS=1,
                DROPOUT=0.1,
                LSTM_BIDIRECTIONAL=True,
                LEARNING_RATE=0.0005,
                PATIENCE=4,
                EPOCHS=2,
                MODEL_PATH=str(Path(p) / "m.pt"),
                MAX_RUL_CYCLE=380000.0,
                MIN_RUL_CYCLE=0.0,
                EARLY_STOP_METRIC="original",
            )
            m = LSTMModel(cfg)
            counter = [0]

            def train(loader, epoch):
                counter[0] = epoch
                with torch.no_grad():
                    next(m.model.parameters()).fill_(epoch + 1)
                return 0.0, np.array([0.2, 0.8]), np.array([0.2, 0.8])

            def validate(loader):
                return (
                    0.0,
                    np.array([0.2, 0.8]) if counter[0] == 0 else np.array([0.9, 0.9]),
                    np.array([0.2, 0.8]),
                )

            m._train_one_epoch = train
            m._validate = validate
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit([], [])
            self.assertTrue(torch.all(next(m.model.parameters()) == 1))

    def test_mc_dropout_restores_module_modes(self):
        m = ConditionTransformer(heads=3, width=8, layers=1, dropout=0.5).eval()
        states = [v.training for v in m.modules()]
        # Make the tiny test heads non-dead independently of random ReLU init.
        with torch.no_grad():
            for head in m.heads:
                head[1].weight.zero_()
                head[1].bias.fill_(1.0)
                head[-1].weight.fill_(1.0)
        _, draws = predict(
            m, np.ones((2, 200, 5), dtype="float32"), torch.device("cpu"), draws=5
        )
        self.assertEqual(draws.shape, (5, 2, 3))
        self.assertGreater(float(draws.std(0).max()), 0.0)
        self.assertEqual([v.training for v in m.modules()], states)


if __name__ == "__main__":
    unittest.main()
