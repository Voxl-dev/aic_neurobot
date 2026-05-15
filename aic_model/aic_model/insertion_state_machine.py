import numpy as np
from enum import Enum, auto


class State(Enum):
    APPROACH = auto()  # moving toward port, no contact yet
    CONTACT  = auto()  # touched something, deciding next action
    EXPLORE  = auto()  # circular search motion looking for port entrance
    SUCCESS  = auto()  # connector seated — task complete
    RETRACT  = auto()  # overforce or timeout — pull back and retry
    FAILED   = auto()  # retries exhausted — give up


class InsertionFSM:
    """F/T-based insertion state machine. Driven by Axia80 at 30 Hz.
    Based on CMU+Intrinsic arXiv:2303.11765."""

    CONTACT_FZ    =  8.0    # N
    # Real bags reached 21.3 N without damage, so 24 N is the safe ceiling.
    SAFETY_FZ     = 24.0    # N
    SAFETY_DUR    =  0.5    # s sustained before RETRACT
    # Calibrated from bag_trial_1/3: dFz=-2.6..-3.9 N, Tz=0.02..0.09 Nm at insertion.
    CLICK_DFZ     = -2.5    # N/frame
    CLICK_TZ      =  0.015  # Nm
    TIMEOUT       = 160.0   # s
    EXPLORE_MAX   =  30.0   # s before giving up and retracting
    MAX_RETRIES   =  3

    EXPLORE_RADIUS = 0.003  # m (3 mm circular search)
    EXPLORE_FREQ   = 0.3    # Hz

    def __init__(self):
        self.state   = State.APPROACH
        self.retries = 0
        self._prev_fz       = 0.0
        self._fz_high_t     = None
        self._explore_start = None
        self._trial_start   = None
        self._t             = 0.0

    def step(self, ft: np.ndarray, t: float, bayesian_confident: bool = True) -> State:
        """Call at 30 Hz. ft = [Fx, Fy, Fz, Tx, Ty, Tz]."""
        if self._trial_start is None:
            self._trial_start = t
        self._t = t

        fz  = float(ft[2])
        tz  = float(ft[5])
        dfz = fz - self._prev_fz
        self._prev_fz = fz

        if t - self._trial_start > self.TIMEOUT:
            self._go(State.FAILED)
            return self.state

        if self.state == State.APPROACH:
            if fz > self.CONTACT_FZ:
                self._go(State.CONTACT)

        elif self.state == State.CONTACT:
            if self._click(dfz, tz):
                self._go(State.SUCCESS)
            elif self._overforce(fz, t):
                self._go(State.RETRACT)
            elif bayesian_confident:
                self._explore_start = t
                self._go(State.EXPLORE)

        elif self.state == State.EXPLORE:
            if self._click(dfz, tz):
                self._go(State.SUCCESS)
            elif self._overforce(fz, t):
                self._go(State.RETRACT)
            elif self._explore_start and (t - self._explore_start) > self.EXPLORE_MAX:
                self._go(State.RETRACT)

        elif self.state == State.RETRACT:
            if self.retries < self.MAX_RETRIES:
                self.retries += 1
                self._fz_high_t = self._explore_start = None
                self._prev_fz = 0.0
                self._go(State.APPROACH)
            else:
                self._go(State.FAILED)

        return self.state

    def _click(self, dfz: float, tz: float) -> bool:
        return dfz < self.CLICK_DFZ and abs(tz) > self.CLICK_TZ

    def _overforce(self, fz: float, t: float) -> bool:
        if fz > self.SAFETY_FZ:
            if self._fz_high_t is None:
                self._fz_high_t = t
            elif t - self._fz_high_t > self.SAFETY_DUR:
                return True
        else:
            self._fz_high_t = None
        return False

    def _go(self, new: State):
        if new != self.state:
            print(f"[FSM] {self.state.name} -> {new.name}  t={self._t:.2f}s")
        self.state = new

    def get_explore_delta(self, t: float) -> np.ndarray:
        """Returns [dx, dy] in meters for the circular search motion."""
        if self.state != State.EXPLORE or self._explore_start is None:
            return np.zeros(2)
        angle = 2 * np.pi * self.EXPLORE_FREQ * (t - self._explore_start)
        return self.EXPLORE_RADIUS * np.array([np.cos(angle), np.sin(angle)])

    @property
    def done(self) -> bool:
        return self.state in (State.SUCCESS, State.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state == State.SUCCESS

    def reset(self):
        self.__init__()
