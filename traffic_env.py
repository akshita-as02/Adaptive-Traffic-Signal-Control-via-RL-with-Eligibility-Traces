"""
traffic_env.py
==============
Simulator for the Adaptive Traffic Signal Control project.
CS 5180 Spring 2026 — Akshita Singh & Pranav Rajesh Nair

MDP Specification (from proposal)
----------------------------------
State  : s = (q_N, q_S, q_E, q_W, phase)
         q_i  in {0,1,2,3}  (binned queue lengths; see BIN_EDGES)
         phase in {0, 1}     (0 = NS green, 1 = EW green)
         |S| = 4^4 * 2 = 512

Action : a in {0=KEEP, 1=SWITCH}
         SWITCH initiates a mandatory YELLOW_STEPS-step yellow delay
         during which the agent's action is ignored.

Reward : R(s,a) = -sum_i q_i(t)   (negative total queue length)
         Bounds: [-4*Q_MAX, 0]

Arrivals: q_i(t) ~ Poisson(lambda_i) each step, independent per lane
Service : one vehicle departs per active GREEN lane per step

Usage
-----
    from traffic_env import TrafficEnv, KEEP, SWITCH

    # Simple variant (2 active lanes, symmetric light load)
    env = TrafficEnv(mode="simple", arrival_rates=[0.3, 0.3], seed=0)

    # Full variant (4 lanes, custom rates)
    env = TrafficEnv(mode="full", arrival_rates=[0.8, 0.8, 0.2, 0.2], seed=0)

    state = env.reset()
    for _ in range(1000):
        action = KEEP                        # or your agent's choice
        next_state, reward, info = env.step(action)

    # Run a built-in baseline for N steps and return mean wait
    from traffic_env import run_baseline
    mean_wait = run_baseline("fixed", env, n_steps=10_000)

Notes for algorithm implementers
---------------------------------
- env.reset()  returns a 5-element numpy array (float32)
- env.step(a)  returns (next_state, reward, info)
  * info["yellow"]   = True if currently in yellow-delay phase
  * info["raw_q"]    = raw (un-binned) queue lengths, shape (n_lanes,)
  * info["phase"]    = current phase after transition
- During yellow, the agent's action is silently ignored; the env
  advances the yellow counter automatically.
- env.state_for_tiling() returns the raw continuous queue lengths
  suitable for passing directly to tile-coding (no binning).
- env.n_lanes is 2 (simple) or 4 (full) — tile-coding calls should
  use this to build feature vectors of the right dimension.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List

# ── Constants ────────────────────────────────────────────────────────────────

KEEP   = 0
SWITCH = 1

# Bin boundaries for discretising raw queue lengths into {0,1,2,3}
# bin 0: 0 vehicles  bin 1: 1-3  bin 2: 4-7  bin 3: 8+
BIN_EDGES = [0, 1, 4, 8]   # right-open: [0,1), [1,4), [4,8), [8,inf)

# Maximum raw queue length stored internally (continuous, not binned)
Q_MAX = 20   # hard cap; effectively infinite relative to bin boundary of 8

# Yellow light duration (steps during which phase does not change)
YELLOW_STEPS = 3


def _bin(q: np.ndarray) -> np.ndarray:
    """Discretise raw queue lengths into bin indices {0,1,2,3}."""
    bins = np.zeros(len(q), dtype=np.int32)
    bins[q >= 1] = 1
    bins[q >= 4] = 2
    bins[q >= 8] = 3
    return bins


# ── Environment ──────────────────────────────────────────────────────────────

class TrafficEnv:
    """
    Single four-way (or two-way) intersection simulator.

    Parameters
    ----------
    mode : "simple" | "full"
        "simple" — 2 lanes (NS, EW), symmetric light traffic.
                   arrival_rates defaults to [0.3, 0.3].
        "full"   — 4 lanes (N, S, E, W).
                   arrival_rates defaults to [0.3, 0.3, 0.3, 0.3].
    arrival_rates : list of float, optional
        Poisson lambda per lane per step.  Length must match mode
        (2 for simple, 4 for full).  Overrides the mode default.
    seed : int or None
        RNG seed for reproducibility.
    yellow_steps : int
        Duration of mandatory yellow-light delay (default 3).
    q_max : int
        Hard cap on raw queue length per lane (default 20).
    """

    # Which lanes get green under each phase
    # simple: lane 0 = NS direction, lane 1 = EW direction
    # full:   lanes 0,1 = N,S (phase 0 green); lanes 2,3 = E,W (phase 1 green)
    _GREEN_LANES = {
        "simple": {0: [0], 1: [1]},
        "full":   {0: [0, 1], 1: [2, 3]},
    }

    def __init__(
        self,
        mode: str = "full",
        arrival_rates: Optional[List[float]] = None,
        seed: Optional[int] = None,
        yellow_steps: int = YELLOW_STEPS,
        q_max: int = Q_MAX,
    ):
        assert mode in ("simple", "full"), \
            f"mode must be 'simple' or 'full', got '{mode}'"
        self.mode = mode
        self.n_lanes = 2 if mode == "simple" else 4
        self.yellow_steps = yellow_steps
        self.q_max = q_max

        # Default arrival rates if not supplied
        if arrival_rates is None:
            arrival_rates = [0.3] * self.n_lanes
        arrival_rates = list(arrival_rates)
        assert len(arrival_rates) == self.n_lanes, (
            f"arrival_rates length {len(arrival_rates)} "
            f"does not match n_lanes={self.n_lanes} for mode='{mode}'"
        )
        assert all(r >= 0 for r in arrival_rates), \
            "All arrival rates must be non-negative"
        self.arrival_rates = np.array(arrival_rates, dtype=np.float64)

        self._green_lanes = self._GREEN_LANES[mode]
        self.rng = np.random.default_rng(seed)

        # Internal state (set by reset)
        self._raw_q: np.ndarray = np.zeros(self.n_lanes, dtype=np.float64)
        self._phase: int = 0
        self._yellow_remaining: int = 0
        self._step_count: int = 0

        # Observation space description (for reference)
        # 5 elements: [q0_bin, q1_bin, (q2_bin, q3_bin if full), phase]
        self.obs_dim = self.n_lanes + 1   # binned queues + phase

    # ── Core API ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset the environment to all-zero queues, phase 0, no yellow.

        Returns
        -------
        state : np.ndarray, shape (n_lanes+1,), dtype float32
            Binned queue lengths followed by current phase.
        """
        self._raw_q = np.zeros(self.n_lanes, dtype=np.float64)
        self._phase = 0
        self._yellow_remaining = 0
        self._step_count = 0
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, Dict]:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        action : int
            KEEP (0) or SWITCH (1).
            Ignored while yellow_remaining > 0.

        Returns
        -------
        next_state : np.ndarray, shape (n_lanes+1,), dtype float32
        reward     : float   — negative total binned queue length
        info       : dict
            "yellow"        : bool  — True if currently in yellow delay
            "raw_q"         : np.ndarray — un-binned queue lengths
            "phase"         : int   — phase after this step
            "yellow_remaining" : int
            "step"          : int
        """
        assert action in (KEEP, SWITCH), \
            f"action must be KEEP(0) or SWITCH(1), got {action}"

        in_yellow = self._yellow_remaining > 0

        # ── Phase / yellow logic ──────────────────────────────────────────
        if in_yellow:
            # Ignore agent action; advance yellow counter
            self._yellow_remaining -= 1
            if self._yellow_remaining == 0:
                # Yellow delay just finished: flip phase
                self._phase = 1 - self._phase
        else:
            if action == SWITCH:
                # Start yellow delay; phase flips after YELLOW_STEPS steps
                self._yellow_remaining = self.yellow_steps
            # else: KEEP — phase unchanged, yellow_remaining stays 0

        # ── Determine which lanes are currently green ────────────────────
        # During yellow, NO lane is served (vehicles wait through yellow)
        if self._yellow_remaining > 0 or in_yellow:
            # We are in yellow (either just started or mid-way).
            # "in_yellow" is True for the step we entered yellow, but we
            # still serve no vehicles on yellow steps.
            green = []
        else:
            green = self._green_lanes[self._phase]

        # ── Service ──────────────────────────────────────────────────────
        for lane in green:
            self._raw_q[lane] = max(0.0, self._raw_q[lane] - 1.0)

        # ── Arrivals ─────────────────────────────────────────────────────
        arrivals = self.rng.poisson(self.arrival_rates)
        self._raw_q = np.minimum(self._raw_q + arrivals, self.q_max)

        # ── Reward ───────────────────────────────────────────────────────
        binned = _bin(self._raw_q)
        reward = -float(np.sum(binned))

        self._step_count += 1

        info = {
            "yellow":            self._yellow_remaining > 0,
            "yellow_remaining":  self._yellow_remaining,
            "raw_q":             self._raw_q.copy(),
            "phase":             self._phase,
            "step":              self._step_count,
        }
        return self._observe(), reward, info

    # ── Observation helpers ───────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        """Return the current discrete observation as float32."""
        binned = _bin(self._raw_q).astype(np.float32)
        obs = np.append(binned, float(self._phase)).astype(np.float32)
        return obs

    def state_for_tiling(self) -> np.ndarray:
        """
        Return raw (un-binned) queue lengths + phase for tile coding.

        Tile coding works best on continuous inputs; this bypasses
        the discretisation so your tiling can create its own bins.

        Returns
        -------
        np.ndarray, shape (n_lanes+1,), dtype float64
            [raw_q_0, ..., raw_q_{n-1}, phase]
        """
        return np.append(self._raw_q.copy(), float(self._phase))

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def in_yellow(self) -> bool:
        return self._yellow_remaining > 0

    # ── State space / discrete index ─────────────────────────────────────

    def obs_to_index(self, obs: np.ndarray) -> int:
        """
        Convert a discrete observation to a flat integer index in [0, 512).
        Useful for tabular methods or debugging lookup tables.

        Index formula: q0*128 + q1*32 + q2*8 + q3*2 + phase  (full)
                       q0*4   + q1*2  + phase                 (simple)
        """
        q = obs[:-1].astype(int)
        ph = int(obs[-1])
        idx = 0
        for qi in q:
            idx = idx * 4 + qi
        idx = idx * 2 + ph
        return idx

    def index_to_obs(self, idx: int) -> np.ndarray:
        """Inverse of obs_to_index. Returns float32 observation."""
        ph = idx % 2
        idx //= 2
        q = []
        for _ in range(self.n_lanes):
            q.append(idx % 4)
            idx //= 4
        q = list(reversed(q))
        return np.array(q + [ph], dtype=np.float32)

    @property
    def n_states(self) -> int:
        return (4 ** self.n_lanes) * 2

    # ── Clone / reproducibility ───────────────────────────────────────────

    def clone_state(self) -> Dict:
        """
        Return a snapshot of the full internal state.
        Useful for saving/restoring mid-episode (e.g., in planning).
        """
        return {
            "raw_q":            self._raw_q.copy(),
            "phase":            self._phase,
            "yellow_remaining": self._yellow_remaining,
            "step_count":       self._step_count,
        }

    def restore_state(self, snapshot: Dict):
        """Restore from a snapshot produced by clone_state()."""
        self._raw_q            = snapshot["raw_q"].copy()
        self._phase            = snapshot["phase"]
        self._yellow_remaining = snapshot["yellow_remaining"]
        self._step_count       = snapshot["step_count"]

    # ── Pretty print ─────────────────────────────────────────────────────

    def render(self):
        """Print a one-line human-readable summary of the current state."""
        q    = self._raw_q
        bq   = _bin(q)
        ph   = "NS-green" if self._phase == 0 else "EW-green"
        yw   = f"  [YELLOW: {self._yellow_remaining} left]" \
               if self._yellow_remaining > 0 else ""
        names = ["N","S","E","W"] if self.mode == "full" else ["NS","EW"]
        lane_str = "  ".join(
            f"{names[i]}:raw={q[i]:.0f}/bin={bq[i]}"
            for i in range(self.n_lanes)
        )
        print(f"step={self._step_count:5d}  {ph}{yw}  |  {lane_str}  "
              f"|  reward={-float(np.sum(bq)):.0f}")

    def __repr__(self):
        return (f"TrafficEnv(mode={self.mode!r}, "
                f"arrival_rates={self.arrival_rates.tolist()}, "
                f"n_lanes={self.n_lanes})")


# ── Baselines ────────────────────────────────────────────────────────────────

def _fixed_timing_policy(env: TrafficEnv, step: int, cycle_len: int = 30) -> int:
    """Alternate phases every cycle_len steps, regardless of queues."""
    phase_step = step % (cycle_len * 2)
    desired_phase = 0 if phase_step < cycle_len else 1
    if desired_phase != env.phase and not env.in_yellow:
        return SWITCH
    return KEEP


def _longest_queue_policy(env: TrafficEnv) -> int:
    """
    Give green to the direction with the highest total raw queue.
    Only switches when the other direction is strictly longer
    and we are not in yellow.
    """
    if env.in_yellow:
        return KEEP
    q = env._raw_q
    if env.mode == "simple":
        ns_load, ew_load = q[0], q[1]
    else:
        ns_load = q[0] + q[1]
        ew_load = q[2] + q[3]

    desired = 0 if ns_load >= ew_load else 1
    if desired != env.phase:
        return SWITCH
    return KEEP


def run_baseline(
    policy: str,
    env: TrafficEnv,
    n_steps: int = 10_000,
    cycle_len: int = 30,
    verbose: bool = False,
) -> Dict:
    """
    Run a non-learning baseline policy for n_steps and collect stats.

    Parameters
    ----------
    policy : "fixed" | "longest_queue"
    env    : TrafficEnv  (will be reset before running)
    n_steps : int
    cycle_len : int  (only used by "fixed" policy)
    verbose : bool  (print render every 1000 steps)

    Returns
    -------
    dict with keys:
        "mean_wait"   : float  — mean per-step reward magnitude
        "total_reward": float
        "rewards"     : list of per-step rewards
    """
    assert policy in ("fixed", "longest_queue"), \
        f"Unknown policy '{policy}'. Choose 'fixed' or 'longest_queue'."

    env.reset()
    rewards = []
    for t in range(n_steps):
        if policy == "fixed":
            action = _fixed_timing_policy(env, t, cycle_len)
        else:
            action = _longest_queue_policy(env)

        _, reward, _ = env.step(action)
        rewards.append(reward)
        if verbose and (t + 1) % 1000 == 0:
            env.render()

    rewards = np.array(rewards)
    return {
        "mean_wait":    float(-rewards.mean()),
        "total_reward": float(rewards.sum()),
        "rewards":      rewards,
    }


# ── Sanity checks (run this file directly to verify) ─────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TrafficEnv Sanity Checks")
    print("=" * 60)

    # ── 1. Basic step mechanics ──────────────────────────────────────────
    print("\n[1] Basic step trace — Simple mode, seed=0")
    env = TrafficEnv(mode="simple", arrival_rates=[0.5, 0.5], seed=0)
    s = env.reset()
    print(f"    Initial obs: {s}")
    for t in range(6):
        action = SWITCH if t == 2 else KEEP
        s, r, info = env.step(action)
        tag = " <-- SWITCH requested" if t == 2 else ""
        print(f"    t={t+1}: action={'SWITCH' if action else 'KEEP ':6s}  "
              f"obs={s}  reward={r:.0f}  yellow={info['yellow']}"
              f"  phase={info['phase']}{tag}")

    # ── 2. Yellow delay correctness ──────────────────────────────────────
    print("\n[2] Yellow delay: phase should flip exactly after YELLOW_STEPS steps")
    env2 = TrafficEnv(mode="simple", arrival_rates=[0.0, 0.0], seed=1)
    env2.reset()
    phases = []
    for t in range(10):
        action = SWITCH if t == 0 else KEEP
        _, _, info = env2.step(action)
        phases.append(info["phase"])
    expected_flip_at = YELLOW_STEPS   # index 3 (0-indexed steps after switch)
    print(f"    Phases after switch at t=0: {phases}")
    print(f"    Expected flip at step {expected_flip_at}: "
          f"{'OK' if phases[expected_flip_at] == 1 else 'FAIL'}")
    assert phases[:YELLOW_STEPS] == [0] * YELLOW_STEPS, "Phase flipped too early!"
    assert phases[YELLOW_STEPS] == 1, "Phase did not flip after yellow delay!"
    print("    Yellow delay logic: PASSED")

    # ── 3. Queue clamping ────────────────────────────────────────────────
    print("\n[3] Queue clamping at q_max")
    env3 = TrafficEnv(mode="simple", arrival_rates=[100.0, 0.0], seed=2,
                      q_max=Q_MAX)
    env3.reset()
    for _ in range(5):
        env3.step(KEEP)
    assert env3._raw_q[0] <= Q_MAX, "Queue exceeded q_max!"
    print(f"    raw_q after flooding lane 0: {env3._raw_q}  (q_max={Q_MAX})  PASSED")

    # ── 4. Reward bounds ─────────────────────────────────────────────────
    print("\n[4] Reward bounds")
    env4 = TrafficEnv(mode="full", arrival_rates=[10.0]*4, seed=3)
    env4.reset()
    rewards = []
    for _ in range(500):
        _, r, _ = env4.step(KEEP)
        rewards.append(r)
    assert all(r >= -12 for r in rewards), "Reward below lower bound!"
    assert all(r <= 0   for r in rewards), "Reward above 0!"
    print(f"    Reward range: [{min(rewards):.0f}, {max(rewards):.0f}]  "
          f"Expected: [-12, 0]  PASSED")

    # ── 5. obs_to_index / index_to_obs round-trip ────────────────────────
    print("\n[5] obs_to_index / index_to_obs round-trip")
    env5 = TrafficEnv(mode="full", seed=42)
    env5.reset()
    all_pass = True
    for idx in range(env5.n_states):
        obs = env5.index_to_obs(idx)
        recovered = env5.obs_to_index(obs)
        if recovered != idx:
            print(f"    FAIL at idx={idx}")
            all_pass = False
    print(f"    All {env5.n_states} states round-trip: {'PASSED' if all_pass else 'FAILED'}")

    # ── 6. Baselines ─────────────────────────────────────────────────────
    print("\n[6] Baseline comparison (Full mode, symmetric lambda=0.4, 10k steps)")
    env6 = TrafficEnv(mode="full", arrival_rates=[0.4]*4, seed=7)
    for name in ("fixed", "longest_queue"):
        stats = run_baseline(name, env6, n_steps=10_000)
        print(f"    {name:15s}  mean_wait = {stats['mean_wait']:.4f}")

    # ── 7. state_for_tiling ──────────────────────────────────────────────
    print("\n[7] state_for_tiling shape and dtype")
    env7 = TrafficEnv(mode="full", seed=0)
    env7.reset()
    env7.step(KEEP)
    tiling_input = env7.state_for_tiling()
    assert tiling_input.shape == (5,), f"Expected shape (5,), got {tiling_input.shape}"
    print(f"    state_for_tiling() = {tiling_input}  shape={tiling_input.shape}  PASSED")

    # ── 8. clone / restore ───────────────────────────────────────────────
    print("\n[8] clone_state / restore_state")
    env8 = TrafficEnv(mode="full", arrival_rates=[0.3]*4, seed=5)
    env8.reset()
    for _ in range(20):
        env8.step(KEEP)
    snap = env8.clone_state()
    obs_before = env8._observe().copy()
    for _ in range(10):
        env8.step(SWITCH)
    env8.restore_state(snap)
    obs_after = env8._observe()
    assert np.array_equal(obs_before, obs_after), "Restore did not recover state!"
    print(f"    State before: {obs_before}")
    print(f"    State after restore: {obs_after}  PASSED")

    print("\n" + "=" * 60)
    print("All sanity checks passed.")
    print("=" * 60)
