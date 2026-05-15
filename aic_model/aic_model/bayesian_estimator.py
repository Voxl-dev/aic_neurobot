"""
Bayesian alignment estimator for the Axia80 wrist force/torque sensor.

This module adapts the memory update used in Ling et al. (arXiv:2502.12514)
to the AIC cable insertion setup:

1. A calibrated linear model converts an Axia80 wrench window into a physical
   measurement: lateral offset X/Y and yaw error Rz.
2. A discrete reliability distribution is kept for each axis.
3. Each new measurement updates the distribution with Bayes' rule.

The estimator intentionally does not depend on ROS. Runtime policies can feed
it either a numpy-like [Fx, Fy, Fz, Tx, Ty, Tz] vector or a WrenchStamped msg.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_CSV = PROJECT_ROOT / "bayesiano" / "calibration_results.csv"
DEFAULT_BAYESIAN_DATA_DIR = PROJECT_ROOT / "bayesiano"
_FLAT_WRENCH_KEYS = (
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
)


@dataclass(frozen=True)
class Calibration:
    """Linear calibration from physical offset to measured wrench.

    The calibration script in this project fits:

        force_or_torque = gain * offset + bias

    where X/Y offsets are in meters and Rz is in radians. During inference we
    invert that equation to estimate the current residual misalignment.
    """

    gain_x: float = -3.105446
    gain_y: float = -36.600301
    gain_rz: float = 0.000841
    bias_x: float = -0.326674
    bias_y: float = -9.523342
    bias_rz: float = -0.004432
    r2_x: float = 0.9309
    r2_y: float = 0.5492
    r2_rz: float = 0.4668

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_CALIBRATION_CSV) -> "Calibration":
        """Load GAIN_* and BIAS_* values exported by tools/calibrate_ft.py."""
        path = Path(path)
        values: dict[str, float] = {}
        r2_values: dict[str, float] = {}

        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = row["param"].strip()
                values[key] = float(row["value"])
                if row.get("r2"):
                    r2_values[key] = float(row["r2"])

        return cls(
            gain_x=values["GAIN_X"],
            gain_y=values["GAIN_Y"],
            gain_rz=values["GAIN_Rz"],
            bias_x=values["BIAS_X"],
            bias_y=values["BIAS_Y"],
            bias_rz=values["BIAS_Rz"],
            r2_x=r2_values.get("GAIN_X", cls.r2_x),
            r2_y=r2_values.get("GAIN_Y", cls.r2_y),
            r2_rz=r2_values.get("GAIN_Rz", cls.r2_rz),
        )

    def wrench_to_offset(self, wrench: Sequence[float]) -> np.ndarray:
        """Convert [Fx, Fy, Fz, Tx, Ty, Tz] into [dx_mm, dy_mm, dRz_deg]."""
        ft = np.asarray(wrench, dtype=float)
        if ft.shape[0] < 6:
            raise ValueError("Axia80 wrench must contain [Fx, Fy, Fz, Tx, Ty, Tz].")

        dx_m = _safe_inverse_linear(ft[0], self.gain_x, self.bias_x)
        dy_m = _safe_inverse_linear(ft[1], self.gain_y, self.bias_y)
        drz_rad = _safe_inverse_linear(ft[5], self.gain_rz, self.bias_rz)
        return np.array([dx_m * 1000.0, dy_m * 1000.0, math.degrees(drz_rad)])


@dataclass(frozen=True)
class AlignmentEstimate:
    """Current estimator output."""

    dx_mm: float
    dy_mm: float
    dRz_deg: float
    covariance: np.ndarray
    reliability: np.ndarray
    confident: bool

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.dx_mm, self.dy_mm, self.dRz_deg], dtype=float)


class DiscreteBayesianAxis:
    """One-dimensional memory module equivalent to the paper's reliability update."""

    def __init__(
        self,
        states: np.ndarray,
        confidence_floor: float = 1e-12,
    ) -> None:
        self.states = np.asarray(states, dtype=float)
        if self.states.ndim != 1 or self.states.size < 3:
            raise ValueError("states must be a 1D grid with at least three values.")
        self.confidence_floor = confidence_floor
        self.probability = np.ones(self.states.size, dtype=float) / self.states.size

    def reset(self) -> None:
        self.probability.fill(1.0 / self.probability.size)

    def predict_after_correction(self, correction: float) -> None:
        """Shift memory after a commanded correction.

        State is residual misalignment. A positive correction is interpreted as
        reducing a positive residual, so residual_new = residual_old - correction.
        """
        shifted = np.interp(
            self.states + correction,
            self.states,
            self.probability,
            left=0.0,
            right=0.0,
        )
        self.probability = _normalize_or_uniform(shifted, self.confidence_floor)

    def update(self, measurement: float, sigma: float) -> None:
        sigma = max(float(sigma), 1e-6)
        likelihood = np.exp(-0.5 * ((self.states - measurement) / sigma) ** 2)
        likelihood = _normalize_or_uniform(likelihood, self.confidence_floor)
        posterior = likelihood * self.probability
        self.probability = _normalize_or_uniform(posterior, self.confidence_floor)

    @property
    def mean(self) -> float:
        return float(np.dot(self.states, self.probability))

    @property
    def variance(self) -> float:
        mu = self.mean
        return float(np.dot((self.states - mu) ** 2, self.probability))

    @property
    def max_reliability(self) -> float:
        return float(np.max(self.probability))


class Axia80BayesianEstimator:
    """Estimate lateral X/Y misalignment and yaw error from Axia80 readings.

    Parameters are intentionally conservative because the current local
    calibration has strong X fit, weaker Y fit, and weak Rz fit. Low R2 values
    inflate measurement uncertainty instead of pretending the signal is precise.
    """

    def __init__(
        self,
        calibration: Calibration | None = None,
        window: int = 15,
        lateral_range_mm: float = 3.0,
        lateral_bin_mm: float = 0.5,
        angular_range_deg: float = 5.0,
        angular_bin_deg: float = 0.5,
        confidence_trace_threshold: float = 1.0,
        reliability_threshold: float = 0.55,
        sensor_noise: Sequence[float] | None = None,
    ) -> None:
        if window < 3:
            raise ValueError("window must be >= 3 samples.")
        self.calibration = calibration or _load_default_calibration()
        self.window = int(window)
        self.history: deque[np.ndarray] = deque(maxlen=self.window)
        self.confidence_trace_threshold = float(confidence_trace_threshold)
        self.reliability_threshold = float(reliability_threshold)

        self.axes = (
            DiscreteBayesianAxis(_symmetric_grid(lateral_range_mm, lateral_bin_mm)),
            DiscreteBayesianAxis(_symmetric_grid(lateral_range_mm, lateral_bin_mm)),
            DiscreteBayesianAxis(_symmetric_grid(angular_range_deg, angular_bin_deg)),
        )
        self.sensor_noise = _coerce_sensor_noise(sensor_noise)
        self._last_raw_measurement = np.zeros(3, dtype=float)

    @classmethod
    def from_project_files(
        cls,
        calibration_csv: str | Path = DEFAULT_CALIBRATION_CSV,
        data_dir: str | Path = DEFAULT_BAYESIAN_DATA_DIR,
        infer_sensor_noise: bool = True,
        **kwargs,
    ) -> "Axia80BayesianEstimator":
        """Create an estimator using the files stored in the bayesiano folder."""
        calibration = Calibration.from_csv(calibration_csv)
        sensor_noise = kwargs.pop("sensor_noise", None)
        if infer_sensor_noise and sensor_noise is None:
            sensor_noise = infer_axia80_noise_from_trials(data_dir)
        return cls(calibration=calibration, sensor_noise=sensor_noise, **kwargs)

    def reset(self) -> None:
        self.history.clear()
        self._last_raw_measurement = np.zeros(3, dtype=float)
        for axis in self.axes:
            axis.reset()

    def predict_after_correction(
        self,
        dx_mm: float = 0.0,
        dy_mm: float = 0.0,
        dRz_deg: float = 0.0,
    ) -> None:
        """Update memory after an external controller applies a correction."""
        for axis, correction in zip(self.axes, (dx_mm, dy_mm, dRz_deg)):
            axis.predict_after_correction(float(correction))

    def update(
        self,
        ft_reading: Sequence[float],
        applied_correction: Sequence[float] | None = None,
    ) -> AlignmentEstimate:
        """Update with one Axia80 reading and return the current estimate.

        ft_reading must be [Fx, Fy, Fz, Tx, Ty, Tz]. If an external controller
        already moved the TCP based on a previous estimate, pass that movement
        as applied_correction=[dx_mm, dy_mm, dRz_deg] so the memory distribution
        is shifted before assimilating the new evidence.
        """
        if applied_correction is not None:
            if len(applied_correction) != 3:
                raise ValueError("applied_correction must be [dx_mm, dy_mm, dRz_deg].")
            self.predict_after_correction(*applied_correction)

        ft = _coerce_ft_reading(ft_reading)
        self.history.append(ft[:6].copy())

        window_ft = np.asarray(self.history, dtype=float)
        mean_ft = np.mean(window_ft, axis=0)
        std_ft = _window_std(window_ft)

        measurement = self.calibration.wrench_to_offset(mean_ft)
        measurement = self._clip_to_model_range(measurement)
        sigma = self._measurement_sigma(std_ft, sample_count=len(window_ft))
        self._last_raw_measurement = measurement

        for axis, value, axis_sigma in zip(self.axes, measurement, sigma):
            axis.update(float(value), float(axis_sigma))

        return self.current

    def update_from_wrench_msg(
        self,
        wrench_msg,
        applied_correction: Sequence[float] | None = None,
    ) -> AlignmentEstimate:
        """Update from geometry_msgs/Wrench or WrenchStamped-like messages."""
        return self.update(wrench_msg_to_array(wrench_msg), applied_correction)

    @property
    def current(self) -> AlignmentEstimate:
        estimate = np.array([axis.mean for axis in self.axes], dtype=float)
        covariance = np.diag([axis.variance for axis in self.axes])
        reliability = np.array([axis.max_reliability for axis in self.axes], dtype=float)
        confident = bool(
            np.trace(covariance) < self.confidence_trace_threshold
            and np.min(reliability) >= self.reliability_threshold
        )
        return AlignmentEstimate(
            dx_mm=float(estimate[0]),
            dy_mm=float(estimate[1]),
            dRz_deg=float(estimate[2]),
            covariance=covariance,
            reliability=reliability,
            confident=confident,
        )

    @property
    def estimate(self) -> np.ndarray:
        return self.current.vector

    @property
    def covariance(self) -> np.ndarray:
        return self.current.covariance.copy()

    @property
    def reliability(self) -> np.ndarray:
        return self.current.reliability.copy()

    @property
    def confident(self) -> bool:
        return self.current.confident

    @property
    def raw_measurement(self) -> np.ndarray:
        """Most recent calibrated measurement before Bayesian smoothing."""
        return self._last_raw_measurement.copy()

    def _measurement_sigma(self, std_ft: np.ndarray, sample_count: int) -> np.ndarray:
        """Map wrench noise and calibration quality into physical uncertainty."""
        n = max(sample_count, 1)
        effective_noise = np.maximum(std_ft[[0, 1, 5]], self.sensor_noise)
        mean_noise = effective_noise / math.sqrt(n)

        sigma_x = abs(_safe_div(mean_noise[0], self.calibration.gain_x)) * 1000.0
        sigma_y = abs(_safe_div(mean_noise[1], self.calibration.gain_y)) * 1000.0
        sigma_rz = abs(math.degrees(_safe_div(mean_noise[2], self.calibration.gain_rz)))

        # Low calibration R2 means the linear model is less trustworthy.
        model_sigma = np.array(
            [
                (1.0 - _bounded_r2(self.calibration.r2_x)) * 1.0,
                (1.0 - _bounded_r2(self.calibration.r2_y)) * 1.5,
                (1.0 - _bounded_r2(self.calibration.r2_rz)) * 2.5,
            ],
            dtype=float,
        )
        bin_floor = np.array([0.25, 0.25, 0.25], dtype=float)
        return np.maximum(np.array([sigma_x, sigma_y, sigma_rz]) + model_sigma, bin_floor)

    def _clip_to_model_range(self, measurement: np.ndarray) -> np.ndarray:
        clipped = measurement.copy()
        for i, axis in enumerate(self.axes):
            clipped[i] = float(np.clip(clipped[i], axis.states[0], axis.states[-1]))
        return clipped


# Backwards-compatible alias for the name used in the project notes.
AxialAlignmentEstimator = Axia80BayesianEstimator


def wrench_msg_to_array(wrench_msg: Any) -> np.ndarray:
    """Convert common Axia80 wrench representations to [Fx, Fy, Fz, Tx, Ty, Tz].

    Accepted inputs:
    - geometry_msgs/WrenchStamped-like objects with `.wrench.force` and `.wrench.torque`.
    - geometry_msgs/Wrench-like objects with `.force` and `.torque`.
    - dictionaries with force_x/force_y/.../torque_z keys.
    - plain sequences or numpy arrays in [Fx, Fy, Fz, Tx, Ty, Tz] order.
    """
    if isinstance(wrench_msg, Mapping):
        return _wrench_mapping_to_array(wrench_msg)

    if isinstance(wrench_msg, np.ndarray) or _is_numeric_sequence(wrench_msg):
        return _coerce_ft_reading(wrench_msg)

    wrench = getattr(wrench_msg, "wrench", wrench_msg)
    force = getattr(wrench, "force", None)
    torque = getattr(wrench, "torque", None)
    if force is None or torque is None:
        raise TypeError(
            "wrench_msg must be WrenchStamped, Wrench, dict, or "
            "[Fx, Fy, Fz, Tx, Ty, Tz]."
        )

    try:
        return np.array(
            [force.x, force.y, force.z, torque.x, torque.y, torque.z],
            dtype=float,
        )
    except AttributeError as exc:
        raise TypeError(
            "wrench_msg force/torque objects must expose x, y and z attributes."
        ) from exc


def infer_axia80_noise_from_trials(
    data_dir: str | Path = DEFAULT_BAYESIAN_DATA_DIR,
    window: int = 15,
) -> np.ndarray:
    """Estimate [Fx, Fy, Tz] short-window noise from bayesiano/bag_trial_* CSVs."""
    data_dir = Path(data_dir)
    window_stds: list[np.ndarray] = []
    for csv_path in sorted(data_dir.glob("bag_trial_*/fts_broadcaster__wrench.csv")):
        rows = _load_wrench_columns(csv_path)
        if rows.shape[0] < window:
            continue
        for start in range(0, rows.shape[0] - window + 1, window):
            chunk = rows[start : start + window]
            window_stds.append(np.std(chunk[:, [0, 1, 5]], axis=0, ddof=1))

    if not window_stds:
        return np.array([0.08, 0.08, 0.01], dtype=float)
    return np.median(np.asarray(window_stds, dtype=float), axis=0)


def fit_linear_calibration(
    ft_data: np.ndarray,
    offsets: np.ndarray,
) -> Calibration:
    """Fit force = gain * offset + bias from paired Axia80 and offset data.

    ft_data:  (N, 6) with columns [Fx, Fy, Fz, Tx, Ty, Tz].
    offsets:  (N, 3) with columns [dx_m, dy_m, dRz_rad].
    """
    ft_data = np.asarray(ft_data, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if ft_data.ndim != 2 or ft_data.shape[1] < 6:
        raise ValueError("ft_data must have shape (N, 6).")
    if offsets.ndim != 2 or offsets.shape[1] != 3 or offsets.shape[0] != ft_data.shape[0]:
        raise ValueError("offsets must have shape (N, 3) and match ft_data rows.")

    gain_x, bias_x, r2_x = _linreg(offsets[:, 0], ft_data[:, 0])
    gain_y, bias_y, r2_y = _linreg(offsets[:, 1], ft_data[:, 1])
    gain_rz, bias_rz, r2_rz = _linreg(offsets[:, 2], ft_data[:, 5])
    return Calibration(gain_x, gain_y, gain_rz, bias_x, bias_y, bias_rz, r2_x, r2_y, r2_rz)


def _load_default_calibration() -> Calibration:
    if DEFAULT_CALIBRATION_CSV.exists():
        return Calibration.from_csv(DEFAULT_CALIBRATION_CSV)
    return Calibration()


def _coerce_sensor_noise(sensor_noise: Sequence[float] | None) -> np.ndarray:
    """Return a finite [std_Fx, std_Fy, std_Tz] noise vector."""
    default_noise = np.array([0.08, 0.08, 0.01], dtype=float)
    if sensor_noise is None:
        return default_noise

    noise = np.asarray(sensor_noise, dtype=float).reshape(-1)
    if noise.size != 3:
        raise ValueError("sensor_noise must contain exactly [std_Fx, std_Fy, std_Tz].")
    if not np.all(np.isfinite(noise)):
        raise ValueError("sensor_noise values must be finite numbers.")
    if np.any(noise < 0.0):
        raise ValueError("sensor_noise values must be non-negative.")
    return noise


def _coerce_ft_reading(ft_reading: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a finite [Fx, Fy, Fz, Tx, Ty, Tz] vector."""
    ft = np.asarray(ft_reading, dtype=float).reshape(-1)
    if ft.size < 6:
        raise ValueError("Axia80 reading must contain [Fx, Fy, Fz, Tx, Ty, Tz].")
    if not np.all(np.isfinite(ft[:6])):
        raise ValueError("Axia80 reading values must be finite numbers.")
    return ft[:6]


def _is_numeric_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _wrench_mapping_to_array(wrench_msg: Mapping[str, Any]) -> np.ndarray:
    if all(key in wrench_msg for key in _FLAT_WRENCH_KEYS):
        return _coerce_ft_reading([wrench_msg[key] for key in _FLAT_WRENCH_KEYS])

    force = wrench_msg.get("force")
    torque = wrench_msg.get("torque")
    if isinstance(force, Mapping) and isinstance(torque, Mapping):
        return _coerce_ft_reading(
            [
                force["x"],
                force["y"],
                force["z"],
                torque["x"],
                torque["y"],
                torque["z"],
            ]
        )

    raise TypeError(
        "wrench dict must contain force_x/force_y/force_z/torque_x/torque_y/"
        "torque_z or nested force/torque dictionaries."
    )


def _load_wrench_columns(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["force_x"]),
                    float(row["force_y"]),
                    float(row["force_z"]),
                    float(row["torque_x"]),
                    float(row["torque_y"]),
                    float(row["torque_z"]),
                ]
            )
    return np.asarray(rows, dtype=float)


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    a = np.column_stack([x, np.ones_like(x)])
    gain, bias = np.linalg.lstsq(a, y, rcond=None)[0]
    pred = gain * x + bias
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(gain), float(bias), float(r2)


def _safe_inverse_linear(value: float, gain: float, bias: float) -> float:
    return _safe_div(value - bias, gain)


def _safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def _bounded_r2(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _symmetric_grid(limit: float, step: float) -> np.ndarray:
    if limit <= 0.0 or step <= 0.0:
        raise ValueError("limit and step must be positive.")
    count = int(round(limit / step))
    return np.linspace(-count * step, count * step, 2 * count + 1)


def _window_std(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1], dtype=float)
    return np.std(values, axis=0, ddof=1)


def _normalize_or_uniform(values: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= floor:
        return np.ones(values.size, dtype=float) / values.size
    return values / total
