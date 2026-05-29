"""Tests for simulation delete / prepare cancellation."""

import pytest

from app.services.simulation_manager import (
    SimulationManager,
    SimulationCancelledError,
)


def test_request_cancel_marks_simulation():
    sim_id = "sim_test_cancel_flag"
    SimulationManager.clear_cancel(sim_id)
    assert not SimulationManager.is_cancelled(sim_id)

    SimulationManager.request_cancel(sim_id)
    assert SimulationManager.is_cancelled(sim_id)

    SimulationManager.clear_cancel(sim_id)
    assert not SimulationManager.is_cancelled(sim_id)


def test_raise_if_cancelled():
    sim_id = "sim_test_cancel_raise"
    SimulationManager.clear_cancel(sim_id)
    manager = SimulationManager()

    manager._raise_if_cancelled(sim_id)

    SimulationManager.request_cancel(sim_id)
    with pytest.raises(SimulationCancelledError):
        manager._raise_if_cancelled(sim_id)

    SimulationManager.clear_cancel(sim_id)
