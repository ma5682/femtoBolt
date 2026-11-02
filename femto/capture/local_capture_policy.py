"""Limits for new local captures, separate from historical view validation."""

# Includes the initial reference. Continuous raw tracking frames are not capped.
MAX_ACCEPTED_VIEWS = 3


def capture_limit_policy():
    return dict(maximum_accepted_views=MAX_ACCEPTED_VIEWS,
                includes_reference_frame=True, raw_tracking_frames_limited=False,
                automatically_finish=True)
