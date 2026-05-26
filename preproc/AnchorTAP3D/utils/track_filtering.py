# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

import numpy as np
import logging

logger = logging.getLogger(__name__)


def filter_outlier_tracks_by_velocity(
    coords: np.ndarray,
    visibs: np.ndarray,
    velocity_threshold: float = 3.0,
    min_valid_frames: int = 3
) -> tuple:
    """
    Filter outlier tracks based on velocity consistency between adjacent frames.

    Args:
        coords: (T, P, 3) array of 3D coordinates
        visibs: (T, P) array of visibility scores
        velocity_threshold: Threshold for velocity change ratio (default: 3.0)
                          Points with velocity change > threshold * median are considered outliers
        min_valid_frames: Minimum number of valid frames required to keep a track (default: 3)

    Returns:
        coords_filtered: (T, P_filtered, 3) array with outlier tracks removed
        visibs_filtered: (T, P_filtered) array with outlier tracks removed
        valid_track_mask: (P,) boolean array indicating which tracks are valid
    """
    T, P, _ = coords.shape

    if P == 0:
        logger.warning("No tracks to filter")
        return coords, visibs, np.array([], dtype=bool)

    if T < 2:
        logger.warning("Not enough frames to compute velocity, returning all tracks")
        return coords, visibs, np.ones(P, dtype=bool)

    # Compute velocities between adjacent frames
    # velocity[t, p] = coords[t+1, p] - coords[t, p]
    velocities = np.diff(coords, axis=0)  # (T-1, P, 3)
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)  # (T-1, P)

    # Compute velocity changes (acceleration magnitude)
    # accel[t, p] = velocity[t+1, p] - velocity[t, p]
    if T >= 3:
        velocity_changes = np.diff(velocities, axis=0)  # (T-2, P, 3)
        velocity_change_magnitudes = np.linalg.norm(velocity_changes, axis=-1)  # (T-2, P)
    else:
        # Not enough frames for acceleration, just use velocity magnitude
        velocity_change_magnitudes = velocity_magnitudes  # (T-1, P)

    # For each track, detect outlier frames based on velocity change
    valid_track_mask = np.ones(P, dtype=bool)

    for p in range(P):
        # Get velocity changes for this track
        vel_changes_p = velocity_change_magnitudes[:, p]  # (T-2,) or (T-1,)

        # Filter out zero velocities (stationary or invisible points)
        nonzero_vel_changes = vel_changes_p[vel_changes_p > 0]

        if len(nonzero_vel_changes) == 0:
            # Track is stationary, keep it
            continue

        # Compute median velocity change for this track
        median_vel_change = np.median(nonzero_vel_changes)

        # Check if there are any extreme outliers
        # An outlier is a frame where velocity change > threshold * median
        outlier_frames = vel_changes_p > (velocity_threshold * median_vel_change)
        num_outliers = outlier_frames.sum()

        # If more than 30% of frames are outliers, mark track as invalid
        outlier_ratio = num_outliers / len(vel_changes_p)
        if outlier_ratio > 0.3:
            valid_track_mask[p] = False
            continue

        # Additional check: if max velocity change is extremely large compared to median
        if len(nonzero_vel_changes) > 0:
            max_vel_change = vel_changes_p.max()
            if max_vel_change > (velocity_threshold * 2 * median_vel_change):
                # Check if this is a consistent pattern or just one spike
                extreme_outliers = vel_changes_p > (velocity_threshold * 2 * median_vel_change)
                if extreme_outliers.sum() >= 2:
                    # Multiple extreme spikes, likely a bad track
                    valid_track_mask[p] = False

    num_filtered = P - valid_track_mask.sum()
    logger.info(f"Filtered {num_filtered}/{P} outlier tracks based on velocity consistency")
    logger.info(f"Kept {valid_track_mask.sum()} valid tracks")

    # Filter coords and visibs
    coords_filtered = coords[:, valid_track_mask, :]
    visibs_filtered = visibs[:, valid_track_mask]

    return coords_filtered, visibs_filtered, valid_track_mask


def filter_outlier_tracks_by_velocity_adaptive(
    coords: np.ndarray,
    visibs: np.ndarray,
    velocity_threshold_percentile: float = 95.0,
    min_valid_frames: int = 3
) -> tuple:
    """
    Filter outlier tracks using adaptive thresholding based on percentiles.
    This is more robust to different scene dynamics.

    Args:
        coords: (T, P, 3) array of 3D coordinates
        visibs: (T, P) array of visibility scores
        velocity_threshold_percentile: Percentile for velocity change threshold (default: 95.0)
                                      Points above this percentile are considered outliers
        min_valid_frames: Minimum number of valid frames required to keep a track

    Returns:
        coords_filtered: (T, P_filtered, 3) array with outlier tracks removed
        visibs_filtered: (T, P_filtered) array with outlier tracks removed
        valid_track_mask: (P,) boolean array indicating which tracks are valid
    """
    T, P, _ = coords.shape

    if P == 0:
        logger.warning("No tracks to filter")
        return coords, visibs, np.array([], dtype=bool)

    if T < 2:
        logger.warning("Not enough frames to compute velocity, returning all tracks")
        return coords, visibs, np.ones(P, dtype=bool)

    # Compute velocities between adjacent frames
    velocities = np.diff(coords, axis=0)  # (T-1, P, 3)
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)  # (T-1, P)

    # Compute velocity changes (acceleration magnitude)
    if T >= 3:
        velocity_changes = np.diff(velocities, axis=0)  # (T-2, P, 3)
        velocity_change_magnitudes = np.linalg.norm(velocity_changes, axis=-1)  # (T-2, P)
    else:
        velocity_change_magnitudes = velocity_magnitudes  # (T-1, P)

    # Compute global threshold based on all tracks
    all_vel_changes = velocity_change_magnitudes.flatten()
    all_vel_changes = all_vel_changes[all_vel_changes > 0]  # Remove zeros

    if len(all_vel_changes) == 0:
        logger.warning("All tracks are stationary, returning all tracks")
        return coords, visibs, np.ones(P, dtype=bool)

    global_threshold = np.percentile(all_vel_changes, velocity_threshold_percentile)
    logger.info(f"Global velocity change threshold ({velocity_threshold_percentile}th percentile): {global_threshold:.4f}")

    # For each track, check validity
    valid_track_mask = np.ones(P, dtype=bool)

    for p in range(P):
        # Get velocity changes for this track
        vel_changes_p = velocity_change_magnitudes[:, p]

        # Count outlier frames (above global threshold)
        outlier_frames = vel_changes_p > global_threshold
        num_outliers = outlier_frames.sum()

        # If more than 40% of frames are outliers, mark track as invalid
        outlier_ratio = num_outliers / len(vel_changes_p)
        if outlier_ratio > 0.4:
            valid_track_mask[p] = False

    num_filtered = P - valid_track_mask.sum()
    logger.info(f"Filtered {num_filtered}/{P} outlier tracks using adaptive threshold")
    logger.info(f"Kept {valid_track_mask.sum()} valid tracks")

    # Filter coords and visibs
    coords_filtered = coords[:, valid_track_mask, :]
    visibs_filtered = visibs[:, valid_track_mask]

    return coords_filtered, visibs_filtered, valid_track_mask
