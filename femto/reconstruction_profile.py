"""Artifact contract for region-first geometry without presentation exports."""

LOCAL_ROUTE_PROFILE = 'local-route-geometry-v1'
LOCAL_ROUTE_SKIP_REASON = 'local_route_has_no_training_export'


def local_route_result(summary, reconstruction):
    """Require matching producer contracts before accepting absent auxiliaries."""
    return (isinstance(summary, dict) and isinstance(reconstruction, dict)
            and summary.get('artifact_profile') == LOCAL_ROUTE_PROFILE
            and reconstruction.get('artifact_profile') == LOCAL_ROUTE_PROFILE
            and reconstruction.get('status') == 'complete_local_geometry'
            and isinstance(reconstruction.get('reconstruction_scope'), dict)
            and type(reconstruction['reconstruction_scope'].get('contributing_frame_count')) is int
            and reconstruction['reconstruction_scope']['contributing_frame_count'] > 0)
