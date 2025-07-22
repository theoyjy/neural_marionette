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
import glob

from model.neural_marionette import NeuralMarionette
from GenerateSkel import draw_skeleton

def read_mesh_verts(path: str):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {path}")
    return np.asarray(mesh.vertices, np.float32)


def voxelize_mesh(mesh_path: str, bmin, blen, res):
    """Mesh → verts → normalise to [-1,1] → occupancy grid.
    Returns **(1,D,H,W)** float32 as expected by NM."""
    verts = read_mesh_verts(mesh_path)
    verts_n = (verts - bmin) / (blen + 1e-5) * 2.0 - 1.0  # [-1,1]
    occ = voxelize(verts_n, (res, res, res), is_binarized=True)  # (1,D,H,W) bool
    occ = torch.from_numpy(occ).float() # (1, 64, 64, 64) float32
    # print('f occ shape:', occ.shape)  # debug
    return occ


def mesh_to_voxel_4ch(mesh_file, bmin, blen, res, grid=64):
    # 1) 读 mesh → 归一化 [-1,1]
    verts = read_mesh_verts(mesh_file)
    verts_n = (verts - bmin) / (blen + 1e-5) * 2.0 - 1.0  # [-1,1]

    # 2) 占用体素  (1,D,H,W)  ← uint8 / bool
    occ_np = voxelize(verts_n, (grid,)*3, True)

    # 3) 拼 4 通道：最简单方式 = 重复
    #    occ_np.squeeze(0) 变 (D,H,W)
    vox_np = np.repeat(occ_np.squeeze(0)[None], 4, 0)   # (4,D,H,W)

    # 4) 回 torch
    return torch.from_numpy(vox_np).float()             # CPU, float32


def marching_cubes_to_mesh(vox, iso=0.5):
    verts, faces, normals, _ = measure.marching_cubes(vox, level=iso)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

def main():
    parser = argparse.ArgumentParser("Neural‑Marionette mesh interpolation")
    parser.add_argument("--folder", required=True)
    # parser.add_argument("--meshT", required=True)
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


    # ---------- voxelise all frames from folder (no channel dim) ----------
    all_pts = []
    all_voxels = []
    obj_files = sorted(glob.glob(os.path.join(args.folder, "*.obj")))
    for i, obj_file in enumerate(obj_files):
        if i > 10:
            break
        print(f"加载 {i+1}/{len(obj_files)}: {os.path.basename(obj_file)}")
        pts0 = read_mesh_verts(obj_file)
        all_pts.append(pts0)

    # operate vstack as before
    all_pts = np.vstack(all_pts)  # (N,3) all vertices from all meshes
    print(f"shape: {all_pts.shape}") 

    bmin = all_pts.min(axis=0)  # (3,) min corner
    blen = (all_pts.max(0) - bmin).max()

    for i, obj_file in enumerate(obj_files):
        if i > 10:
            break
        print(f"处理 {i+1}/{len(obj_files)}: {os.path.basename(obj_file)}")
        vox = voxelize_mesh(obj_file, bmin, blen, args.res) # (1, 64, 64, 64))
        # vox = mesh_to_voxel_4ch(obj_file, bmin, blen, args.res)
        all_voxels.append(vox) 

    # stack along T → torch & to device (like dataset format)
    print(f"堆叠所有voxel帧...")

    # voxel_seq_t = torch.stack(all_voxels, 0)        # (T,4,64,64,64)
    # voxel_seq_t = voxel_seq_t.float().to(device)


    voxel_seq_t = torch.stack(all_voxels, dim=0).to(device)   # (N, 1, 64, 64, 64)
    voxel_seq_t = voxel_seq_t.unsqueeze(0)  # (1, N, 1, 64, 64, 64)

    with torch.no_grad():
        # kypt_detector expects (B, T, C, H, W, D) format
        detector_log = network.kypt_detector(voxel_seq_t)
        keypoints     = detector_log['keypoints']         # (1, T, K, 4)
        first_feature = detector_log['first_feature']     # (1, C, G, G, G)

        B, T, K, _ = keypoints.shape
        T_total = args.n + 2

         # 获取父节点关系
        parents = network.dyna_module.parents.cpu().numpy()
        
        # 对每一帧进行解码以获得详细的骨骼信息
        for t in range(T):

            # 构造当前帧的keypoints
            selected = torch.zeros(B, 2, K, 4, device=device, dtype=keypoints.dtype)
            selected[:, 0] = keypoints[:, t]     # 当前帧
            selected[:, 1] = keypoints[:, min(t+1, T-1)]  # 下一帧或最后一帧
            selected[:, :, :, 3] = keypoints[:, t, :, 3].unsqueeze(1)  # 保持置信度
            
            first_frame = voxel_seq_t[t:t+1].unsqueeze(0)  # (1, 1, D, H, W)
            
            # 解码获得更详细的骨骼信息
            decode_log = network.kypt_detector.decode_from_dyna(selected, first_feature, first_frame)
            gen_vox = decode_log['gen'].squeeze(0).squeeze(1)       # -> (T_total-2, D, H, W)

            # 提取当前帧的骨骼信息
            frame_kps = keypoints[0, t].cpu().numpy()  # (K, 4)
            joints = frame_kps[:, :3]  # (K, 3) joint positions
            
            # 获取旋转信息
            affinity = detector_log['affinity'] if 'affinity' in detector_log else None
            if affinity is not None:
                R = decode_log['R'][0, 0].cpu().numpy()  # (K, 3, 3)
            else:
                # 如果没有affinity，使用单位旋转矩阵
                R = np.tile(np.eye(3), (K, 1, 1))

            # 画骨架
            gem = draw_skeleton(
                joints, parents, 
                out_path=os.path.join(args.output_dir, f"frame_{t:03d}.png"),
                title=f"Frame {t+1} Skeleton"
            )
            o3d.io.visualization.draw_geometries([gem], window_name=f"Frame {t+1} Skeleton")


    # ---------- concatenate full sequence ----------
    # Remove batch dimension for processing
    voxel_seq_t = voxel_seq_t.squeeze(0)  # (N, C, D, H, W)
    if voxel_seq_t.dim() == 5:
        voxel_seq_t = voxel_seq_t.squeeze(1)  # Remove channel dim if needed
    
    # gen_vox should be (T_total-2, D, H, W), remove batch and channel dims if present
    if gen_vox.dim() > 4:
        gen_vox = gen_vox.squeeze(0).squeeze(1)
    
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
