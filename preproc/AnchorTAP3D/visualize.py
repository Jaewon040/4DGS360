# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

import os
import numpy as np
import cv2
import json
import struct
import zlib
import argparse
from einops import rearrange
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import http.server
import socketserver
import socket
import sys
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer

viz_html_path = Path(__file__).parent / "utils" / "viz.html"
DEFAULT_PORT = 8000

# Set to True to disable first-frame normalization.
# In this codebase extrinsics are W2C and tracks are in world coords (see inference.py),
# so the existing normalization (written for TAPIP3D's C2W convention) misaligns the data.
NO_NORM = True

def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def process_point_cloud_data(input_path, output_file, width=256, height=192, fps=4, query_time=0):
    fixed_size = (width, height)

    input_path = Path(input_path)

    # Check if input is npz file or directory with npy files
    if input_path.suffix == '.npz':
        data = np.load(input_path)
        extrinsics = data["extrinsics"]
        intrinsics = data["intrinsics"]
        trajs = data["coords"]
        video = data["video"]
        depths = data["depths"]
        visibilities = None  # npz files don't have visibility info
    else:
        # Load from npy files
        tapip3d_dir = input_path / "Tapip3d"

        # Load video, depths, intrinsics, extrinsics
        video = np.load(tapip3d_dir / "vid" / "1x" / "video.npy")
        depths = np.load(tapip3d_dir / "depth" / "1x" / "depths.npy")
        intrinsics = np.load(tapip3d_dir / "intrinsic" / "intrinsics.npy")
        extrinsics = np.load(tapip3d_dir / "extrinsic" / "extrinsics.npy")

        # Load trajectory for specific query time
        tracks_dir = tapip3d_dir / "tracks"
        track_files = sorted(tracks_dir.glob("track3d_*.npy"))

        if query_time >= len(track_files):
            raise ValueError(f"Query time {query_time} out of range (0-{len(track_files)-1})")

        track_file = track_files[query_time]
        track_data = np.load(track_file)  # T x P x 4 (xyz + visibility)
        trajs = track_data[:, :, :3]  # T x P x 3
        visibilities = track_data[:, :, 3]  # T x P (boolean)

    T, C, H, W = video.shape
    
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (W / fx) / (H / fy)
    
    rgb_video = (rearrange(video, "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])

    depth_video = depths.astype(np.float32)
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])
    
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5
    
    depth_normalized = (depth_video - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)

    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)

    # Normalize to first frame coordinate system (or skip if NO_NORM is True)
    if NO_NORM:
        normalized_extrinsics = extrinsics.copy()
        normalized_trajs = trajs.copy()
    else:
        first_frame_inv = np.linalg.inv(extrinsics[0])
        normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])

        normalized_trajs = np.zeros_like(trajs)
        for t in range(T):
            homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
            transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
            normalized_trajs[t] = transformed_trajs[:, :3]
    
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }

    # Add visibility if available
    if visibilities is not None:
        arrays["visibilities"] = visibilities.astype(np.uint8)  # boolean -> uint8 for binary format
    
    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)
    
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": fixed_size,
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }
    
    compress_and_write(output_file, header, compressed_blob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the input .result.npz file or directory with npy files')
    parser.add_argument('--width', '-W', type=int, default=256, help='Target width')
    parser.add_argument('--height', '-H', type=int, default=192, help='Target height')
    parser.add_argument('--fps', type=int, default=4, help='Base frame rate for playback')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'Port to serve the visualization (default: {DEFAULT_PORT})')
    parser.add_argument('--query_time', '-q', type=int, default=0, help='Query frame index to visualize (default: 0)')

    args = parser.parse_args()

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        process_point_cloud_data(
            args.input_file,
            temp_path / "data.bin",
            width=args.width,
            height=args.height,
            fps=args.fps,
            query_time=args.query_time
        )
        shutil.copy(viz_html_path, temp_path / "index.html")
        
        os.chdir(temp_path)
        
        host = "127.0.0.1"
        port = args.port
        
        Handler = SimpleHTTPRequestHandler
        httpd = None

        try:
            httpd = ThreadingTCPServer((host, port), Handler)
        except OSError as e:
            if e.errno == socket.errno.EADDRINUSE:
                print(f"Port {port} is already in use, trying a random port...")
                try:
                    httpd = ThreadingTCPServer((host, 0), Handler)
                    port = httpd.server_address[1] # Get the assigned port
                except OSError as e2:
                    print(f"Failed to bind to a random port: {e2}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Failed to start server: {e}", file=sys.stderr)
                sys.exit(1)
        
        if httpd:
            print(f"Serving at http://{host}:{port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
            finally:
                httpd.server_close()

if __name__ == "__main__":
    main()