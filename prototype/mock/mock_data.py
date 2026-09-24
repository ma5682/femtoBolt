"""Compatibility imports; state models now live outside mock providers."""
from models.task_state import Stroke
from app.app_state import AppState as MockState
from mock.mock_capture import SCENARIOS
