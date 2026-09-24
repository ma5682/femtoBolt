"""Compatibility wrapper for the fixed bottom action bar."""

from widgets.action_bar import ActionBar


class BottomBar(ActionBar):
    """Expose the existing action bar under the product-specification name."""

    pass


PrimaryActionBar = BottomBar

__all__ = ["BottomBar", "PrimaryActionBar", "ActionBar"]
