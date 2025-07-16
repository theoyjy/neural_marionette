"""
Given two meshes (first & last frame of the *same* volumetric‑video clip),
this script uses **Neural Marionette** to synthesise T in‑between frames by
leveraging the model’s latent dynamics → voxel decoder, then converts each
voxel frame back to a triangle mesh via marching‑cubes.

Key pipeline
  1.  **Global normalisation**   – both meshes share one bounding‑box → [-1,1]^3.
  2.  **Voxelise** each normalised mesh to 64³ occupancy.
  3.  **Keypoint detect** (kypt_detector) on the two voxels.
  4.  Build a (K × T × 4)  tensor `selected_keypoints` with only first & last
      frames filled; the decoder fills the blanks.
  5.  **decode_from_dyna** → voxel sequence.
  6.  Marching‑cubes → *.ply* mesh per frame (in *world* coords).

Usage
─────
$ python interp_two_meshes.py \
          --mesh0 path/to/frame0.obj \
          --meshT path/to/frameT.obj \
          --ckpt  path/to/marionette.pt \
          --num_frames 10 \
          --outdir output_dir

The script writes 12 meshes: *frame_000.ply* … *frame_011.ply* (first+last+10 in‑betweens).
"""
import os, argparse, os, pickle, time
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import trimesh
from skimage import measure
from utils.dataset_utils import voxelize    # repo util – returns (D,H,W) boolean array

from model.neural_marionette import NeuralMarionette

def read_mesh_verts(path: str):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {path}")
    return np.asarray(mesh.vertices, np.float32)


def voxelize_mesh(mesh_path: str, bmin, blen, res):
    """Mesh → verts → normalise to [-1,1] → occupancy grid.
    Returns **(D,H,W)** float32 without channel dim (NM expects this)."""
    verts = read_mesh_verts(mesh_path)
    verts_n = (verts - bmin) / (blen + 1e-5) * 2.0 - 1.0  # [-1,1]
    occ = voxelize(verts_n, (res, res, res), is_binarized=True)  # (D,H,W) bool
    return occ.astype(np.float32)


def marching_cubes_to_mesh(vox, iso=0.5):
    verts, faces, normals, _ = measure.marching_cubes(vox, level=iso)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

def main():
    parser = argparse.ArgumentParser("Neural‑Marionette mesh interpolation")
    parser.add_argument("--mesh0", required=True)
    parser.add_argument("--meshT", required=True)
    parser.add_argument("--exp_dir", default="pretrained/aist", help="dir with opt.pickle & *.pth")
    parser.add_argument("-n", "--n", type=int, default=10, help="#in‑between frames")
    parser.add_argument("--res", type=int, default=64, help="voxel resolution")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="output/blend", help="output directory base")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- load NM checkpoint ----------
    opt_path = Path(args.exp_dir) / "opt.pickle"
    ckpt_path = next(Path(args.exp_dir).glob("*.pth"))
    with open(opt_path, "rb") as f:
        opt = pickle.load(f)
    network = NeuralMarionette(opt).to(device).eval()
    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.anneal(1)

    # ---------- shared normalisation ----------
    pts0 = read_mesh_verts(args.mesh0)
    ptsT = read_mesh_verts(args.meshT)
    all_pts = np.vstack([pts0, ptsT])
    bmin = all_pts.min(0)
    blen = (all_pts.max(0) - bmin).max()

    # ---------- voxelise two frames (no channel dim) ----------
    vox0 = voxelize_mesh(args.mesh0, bmin, blen, args.res)  # (D,H,W)
    voxT = voxelize_mesh(args.meshT, bmin, blen, args.res)

    # stack along T (2,D,H,W) → torch & to device
    voxel_seq_t = torch.from_numpy(np.stack([vox0, voxT], 0)).to(device)  # float32
    
    print("voxel_seq_t shape", tuple(voxel_seq_t.shape))  # debug

    with torch.no_grad():
        # kypt_detector expects (B,T,D,H,W)
        detector_log = network.kypt_detector(voxel_seq_t.unsqueeze(0))  # (1,2,D,H,W)
        keypoints     = detector_log['keypoints']         # (1,2,K,4)
        first_feature = detector_log['first_feature']     # (1,C,G,G,G)

        B, _, K, _ = keypoints.shape
        T_total = args.n + 2
        selected = torch.zeros(B, T_total, K, 4, device=device, dtype=keypoints.dtype)
        selected[:, 0]  = keypoints[:, 0]
        selected[:, -1] = keypoints[:, 1]
        selected[:, :, :, 3] = keypoints[:, 0, :, 3].unsqueeze(1)

        first_frame = voxel_seq_t.unsqueeze(0)[:, 0]      # (1,D,H,W)

        decode_log = network.kypt_detector.decode_from_dyna(selected, first_feature, first_frame)
        gen_vox = decode_log['gen'].squeeze(0).squeeze(1)       # -> (T_total-2, D, H, W)

    # ---------- concatenate full sequence ----------
    if voxel_seq_t.dim() == 5:
        voxel_seq_t = voxel_seq_t.squeeze(1)
    vox_full = torch.cat([voxel_seq_t[0:1], gen_vox, voxel_seq_t[1:2]], 0)  # (T_total,D,H,W)
    vox_full_np = vox_full.cpu().numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, occ in enumerate(vox_full_np):
        mesh_mc = marching_cubes_to_mesh(occ)
        mesh_mc.vertices = mesh_mc.vertices / (args.res - 1) * blen + bmin
        mesh_mc.export(out_dir / f"frame_{t:03d}.ply")
        print(f"saved frame_{t:03d}.ply")

    print("Done. Interpolated meshes are in", out_dir)


if __name__ == "__main__":
    main()
