"""Tests for EMAEncoder."""

import copy
import math
import pytest
import torch
import torch.nn as nn

from RAG_supporters.nn.models.ema_encoder import EMAEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_encoder():
    torch.manual_seed(0)
    return nn.Linear(16, 16)


@pytest.fixture
def ema(small_encoder):
    return EMAEncoder(small_encoder, tau_min=0.9, tau_max=0.99)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(4, 16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_target_copy(self, ema, small_encoder):
        """Target encoder must be a distinct object with equal initial weights."""
        assert ema.target_encoder is not ema.online_encoder, \
            "Target encoder must be a separate object from online encoder"
        for op, tp in zip(ema.online_encoder.parameters(), ema.target_encoder.parameters()):
            assert torch.equal(op.data, tp.data), \
                "Target and online encoder should start with identical weights"

    def test_target_requires_no_grad(self, ema):
        for p in ema.target_encoder.parameters():
            assert not p.requires_grad, "Target encoder parameters must be frozen (requires_grad=False)"

    def test_online_requires_grad(self, ema):
        for p in ema.online_encoder.parameters():
            assert p.requires_grad, "Online encoder parameters must be trainable (requires_grad=True)"

    def test_invalid_tau_raises(self, small_encoder):
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.99, tau_max=0.9)   # min > max
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.0, tau_max=0.9)    # min == 0
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.9, tau_max=1.0)    # max == 1

    def test_invalid_schedule_raises(self, small_encoder):
        with pytest.raises(ValueError, match="Unsupported schedule"):
            EMAEncoder(small_encoder, schedule="linear")


# ---------------------------------------------------------------------------
# Tau schedule
# ---------------------------------------------------------------------------


class TestTauSchedule:
    def test_tau_at_step_zero(self, ema):
        tau = ema.get_tau(step=0, max_steps=100)
        # At step 0 progress=0 → cosine at 0 gives tau_min
        assert abs(tau - ema.tau_min) < 1e-6, \
            f"tau at step 0 should equal tau_min={ema.tau_min}, got {tau}"

    def test_tau_at_last_step(self, ema):
        tau = ema.get_tau(step=100, max_steps=100)
        assert abs(tau - ema.tau_max) < 1e-6, \
            f"tau at final step should equal tau_max={ema.tau_max}, got {tau}"

    def test_tau_monotonically_increasing(self, ema):
        taus = [ema.get_tau(s, 100) for s in range(0, 101, 10)]
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1] + 1e-9, f"tau not increasing at step {i*10}"

    def test_tau_within_bounds(self, ema):
        for step in range(0, 110, 10):
            tau = ema.get_tau(step, 100)
            assert ema.tau_min - 1e-7 <= tau <= ema.tau_max + 1e-7, \
                f"tau={tau} at step={step} is outside [{ema.tau_min}, {ema.tau_max}]"

    def test_tau_clamps_beyond_max_steps(self, ema):
        tau = ema.get_tau(step=9999, max_steps=100)
        assert abs(tau - ema.tau_max) < 1e-6, \
            f"tau beyond max_steps should clamp to tau_max={ema.tau_max}, got {tau}"

    def test_zero_max_steps_returns_tau_max(self, ema):
        tau = ema.get_tau(0, 0)
        assert tau == ema.tau_max, \
            f"get_tau with max_steps=0 should return tau_max={ema.tau_max}, got {tau}"

    def test_tau_info_dict(self, ema):
        info = ema.get_tau_info(10, 100)
        assert "tau" in info, "get_tau_info should include 'tau' key"
        assert info["tau_min"] == ema.tau_min, \
            f"tau_info['tau_min'] should be {ema.tau_min}, got {info.get('tau_min')}"
        assert info["tau_max"] == ema.tau_max, \
            f"tau_info['tau_max'] should be {ema.tau_max}, got {info.get('tau_max')}"


# ---------------------------------------------------------------------------
# EMA update mechanics
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_target_params_change_after_update(self, ema):
        original_target = {
            name: param.data.clone()
            for name, param in ema.target_encoder.named_parameters()
        }
        # Simulate one gradient step on online encoder
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data += 0.1

        ema.update_target(step=0, max_steps=100)

        for name, param in ema.target_encoder.named_parameters():
            assert not torch.equal(param.data, original_target[name]), \
                f"Target param '{name}' should have changed after EMA update"

    def test_target_drifts_slower_than_online(self, small_encoder):
        """Target params should change less than online params per step."""
        ema = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.99, tau_max=0.999)
        initial_online = {n: p.data.clone() for n, p in ema.online_encoder.named_parameters()}
        initial_target = {n: p.data.clone() for n, p in ema.target_encoder.named_parameters()}

        # Large update to online encoder
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data += 1.0

        ema.update_target(0, 100)

        for name in initial_online:
            online_delta = (ema.online_encoder.state_dict()[name] - initial_online[name]).abs().mean()
            target_delta = (ema.target_encoder.state_dict()[name] - initial_target[name]).abs().mean()
            assert target_delta.item() < online_delta.item(), (
                f"Target ({target_delta:.4f}) drifted more than online ({online_delta:.4f}) for {name}"
            )

    def test_update_does_not_restore_grad_to_target(self, ema):
        ema.update_target(0, 100)
        for p in ema.target_encoder.parameters():
            assert not p.requires_grad, \
                "EMA update must not re-enable gradients on target encoder parameters"

    def test_multiple_updates_converge(self, small_encoder):
        """After many updates the target should approach the online encoder."""
        ema = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.5, tau_max=0.5)
        # Move online encoder far away
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data.fill_(10.0)
        # Run many EMA updates
        for step in range(200):
            ema.update_target(step, 200)

        for op, tp in zip(ema.online_encoder.parameters(), ema.target_encoder.parameters()):
            diff = (op.data - tp.data).abs().mean()
            assert diff.item() < 1.0, f"Target not converging: diff={diff:.4f}"


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


class TestForward:
    def test_forward_uses_online_encoder(self, ema, x):
        out_ema = ema(x)
        out_online = ema.online_encoder(x)
        assert torch.allclose(out_ema, out_online), \
            "EMAEncoder.forward() should route through the online encoder"

    def test_encode_target_uses_target_encoder(self, ema, x):
        out = ema.encode_target(x)
        expected = ema.target_encoder(x)
        assert torch.allclose(out, expected), \
            "encode_target() output must match target_encoder(x) directly"

    def test_encode_target_returns_detached(self, ema, x):
        out = ema.encode_target(x)
        assert not out.requires_grad, \
            "encode_target() must return a detached tensor (no gradient)"

    def test_online_forward_produces_grad(self, ema, x):
        x_grad = x.requires_grad_(True)
        out = ema(x_grad)
        assert out.requires_grad, \
            "EMAEncoder.forward() should produce a grad-enabled output for training"

    def test_target_does_not_participate_in_backprop(self, ema, x):
        """No grad should accumulate in target params."""
        out = ema.encode_target(x)
        # Trying to backward should error if grad flows; otherwise no grad set
        for p in ema.target_encoder.parameters():
            assert p.grad is None, \
                "Target encoder parameters must not accumulate gradients"


# ---------------------------------------------------------------------------
# State dict save / load
# ---------------------------------------------------------------------------


class TestStateDict:
    def test_state_dict_contains_tau_keys(self, ema):
        sd = ema.state_dict()
        assert "_ema_tau_min" in sd, "state_dict must contain '_ema_tau_min'"
        assert "_ema_tau_max" in sd, "state_dict must contain '_ema_tau_max'"
        assert "_ema_schedule" in sd, "state_dict must contain '_ema_schedule'"

    def test_state_dict_round_trip(self, small_encoder):
        ema1 = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.95, tau_max=0.98)
        # Modify online encoder
        with torch.no_grad():
            for p in ema1.online_encoder.parameters():
                p.data += 0.5
        ema1.update_target(10, 100)

        sd = ema1.state_dict()

        ema2 = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.0001, tau_max=0.9999)
        ema2.load_state_dict(sd)

        assert ema2.tau_min == ema1.tau_min, \
            "Loaded tau_min should match saved tau_min"
        assert ema2.tau_max == ema1.tau_max, \
            "Loaded tau_max should match saved tau_max"

        for p1, p2 in zip(ema1.online_encoder.parameters(), ema2.online_encoder.parameters()):
            assert torch.equal(p1.data, p2.data), \
                "Loaded online encoder weights should match saved weights"

        for p1, p2 in zip(ema1.target_encoder.parameters(), ema2.target_encoder.parameters()):
            assert torch.equal(p1.data, p2.data), \
                "Loaded target encoder weights should match saved weights"

    def test_load_restores_frozen_target(self, small_encoder, ema):
        sd = ema.state_dict()
        ema2 = EMAEncoder(copy.deepcopy(small_encoder))
        ema2.load_state_dict(sd)
        for p in ema2.target_encoder.parameters():
            assert not p.requires_grad, \
                "load_state_dict must keep target encoder parameters frozen"

    def test_repr(self, ema):
        r = repr(ema)
        assert "EMAEncoder" in r, "repr should contain 'EMAEncoder'"
        assert str(ema.tau_min) in r, "repr should contain the tau_min value"
