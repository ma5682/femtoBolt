"""Compatibility page name for the final planning-preview confirmation."""

from pages.confirmation_page import ConfirmationPage


class ConfirmPage(ConfirmationPage):
    """Current confirmation page exposed as ``ConfirmPage`` for new shells."""

    pass


__all__ = ["ConfirmPage", "ConfirmationPage"]
