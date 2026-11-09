"""Limits for new local captures, separate from historical view validation."""

# New acquisitions stop after three durably accepted views, including W0.
MAX_ACCEPTED_VIEWS = 3


def capture_limit_policy(*, region_first=False):
    return dict(maximum_accepted_views=MAX_ACCEPTED_VIEWS,
                minimum_accepted_views=1 if region_first else 3,
                includes_reference_frame=True, raw_tracking_frames_limited=False,
                automatically_finish=True,
                finish_mode='manual_or_view_limit' if region_first else 'view_limit',
                session_duration_limit_s=300)


def view_limit_reached(count, *, region_first=False):
    return count >= MAX_ACCEPTED_VIEWS
