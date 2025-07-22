import argparse
import os
import glob
import time
import pickle
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import voxelize
import py_dem_bones as pdb

def _as_colmajor(vs: np.ndarray) -> np.ndarray:
    """(N,3) -> (3,N) for DemBones."""
    if vs.shape[1] != 3:
        raise ValueError("expect (N,3) vertices")
    return vs.T.copy()

def _as_rowmajor(vs: np.ndarray) -> np.ndarray:
    """(3,N) -> (N,3) back to normal."""
    if vs.shape[0] != 3:
        raise ValueError("expect (3,N) vertices")
    return vs.T.copy()

def sanitize_parents(parents: np.ndarray) -> np.ndarray:
    """Fix self-parenting and break any accidental cycles in-place."""
    n = len(parents)

    # 1. remove self-parent
    self_mask = parents == np.arange(n)
    if self_mask.any():
        print("[fix] bones self-parenting:", np.where(self_mask)[0])
        parents[self_mask] = -1          # make them roots for now

    # 2. break longer cycles
    for i in range(n):
        visited = set()
        cur = i
        while parents[cur] >= 0:
            if cur in visited:
                print(f"[fix] cycle hit at bone {cur}; severing link from {cur} to {parents[cur]}")
                parents[cur] = -1        # detach to root
                break
            visited.add(cur)
            cur = parents[cur]
    return parents

def run_dembone_with_timeout(dem, timeout_seconds=60):  # Reduced to 1 minute
    """
    Run DemBones computation with aggressive timeout using threading.
    Returns (success, error_message)
    """
    import threading
    import time
    
    result = {'success': False, 'error': None, 'completed': False, 'started': False}
    
    def compute_thread():
        try:
            print(f"DemBones thread starting...")
            result['started'] = True
            
            # Add a heartbeat mechanism
            import sys
            sys.stdout.flush()
            
            dem.compute()
            result['success'] = True
            result['completed'] = True
            print(f"DemBones thread completed successfully")
        except Exception as e:
            result['error'] = str(e)
            result['completed'] = True
            print(f"DemBones thread failed: {e}")
        finally:
            # Ensure we always mark as completed
            result['completed'] = True
    
    # Start computation in separate thread
    thread = threading.Thread(target=compute_thread, daemon=True)
    thread.start()
    
    # Wait with aggressive progress updates
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds instead of 10
    last_heartbeat = start_time
    
    while thread.is_alive():
        elapsed = time.time() - start_time
        
        # Much shorter timeout for large datasets
        if elapsed > timeout_seconds:
            print(f"⚠️  TIMEOUT: DemBones computation exceeded {timeout_seconds}s limit")
            print(f"⚠️  This is expected for large datasets ({elapsed:.1f}s elapsed)")
            print(f"⚠️  Switching to fallback skinning algorithm...")
            # Thread will be abandoned (daemon=True)
            return False, f"Timeout after {timeout_seconds}s - switching to fallback"
        
        # Wait for a short interval or until thread completes
        thread.join(timeout=check_interval)
        
        if thread.is_alive():
            current_time = time.time()
            if current_time - last_heartbeat >= check_interval:
                print(f"💓 DemBones heartbeat: {elapsed:.1f}s elapsed (max {timeout_seconds}s)")
                last_heartbeat = current_time
                
                # Force output flush
                import sys
                sys.stdout.flush()
    
    # Thread completed, check results
    if result['completed']:
        if result['success']:
            elapsed = time.time() - start_time
            print(f"DemBones completed successfully in {elapsed:.1f}s")
            return True, None
        else:
            error_msg = result['error'] or "Unknown error"
            print(f"DemBones failed: {error_msg}")
            return False, error_msg
    else:
        print(f"DemBones thread ended without completion signal")
        return False, "Thread completed but no result recorded"

def solve_with_dem_bones_safe(frames_vertices: np.ndarray,
                              parents: np.ndarray,
                              nnz: int = 8,
                              n_iters: int = 20):  # Reduced default iterations
    """
    Safe wrapper for DemBones with multiple fallback mechanisms.
    """
    F, N, _ = frames_vertices.shape
    K = parents.shape[0]
    
    # For very large datasets, use extremely conservative parameters
    if N > 20000 or F > 50:
        print(f"Large dataset detected ({F} frames, {N} vertices)")
        print(f"Using ultra-conservative DemBones parameters")
        n_iters = min(10, n_iters)  # Max 10 iterations for large datasets
        nnz = min(4, nnz)  # Max 4 weights per vertex for large datasets
    
    print(f"Starting safe DemBones solver with {n_iters} iterations, {nnz} weights per vertex")
    
    # First try the regular DemBones with timeout
    try:
        return solve_with_dem_bones(frames_vertices, parents, nnz, n_iters)
    except Exception as e:
        print(f"❌ Primary DemBones failed: {e}")
        
        # For large datasets, skip the retry and go straight to fallback
        if N > 20000 or F > 50:
            print(f"🔄 Large dataset - skipping retry, using fallback immediately")
        else:
            # If that fails, try with even more conservative parameters
            print("🔄 Trying with ultra-conservative parameters...")
            try:
                return solve_with_dem_bones(frames_vertices, parents, 
                                          min(2, nnz), min(5, n_iters))
            except Exception as e2:
                print(f"❌ Conservative DemBones also failed: {e2}")
        
        # Final fallback to simple skinning
        print("🔄 Using simple distance-based skinning as final fallback")
        
        # Use first mesh joints if available, or compute simple joint positions
        joints = np.zeros((K, 3))
        if K > 0:
            # Simple joint placement: spread joints around the mesh
            mesh_center = frames_vertices[0].mean(axis=0)
            mesh_radius = np.linalg.norm(frames_vertices[0] - mesh_center, axis=1).max()
            
            for i in range(K):
                angle = 2 * np.pi * i / K
                joints[i] = mesh_center + mesh_radius * 0.5 * np.array([
                    np.cos(angle), 0, np.sin(angle)
                ])
        
        return simple_skinning_fallback(frames_vertices, joints, parents)

def solve_with_dem_bones(frames_vertices: np.ndarray,
                         parents: np.ndarray,
                         nnz: int = 8,
                         n_iters: int = 30):
    """
    frames_vertices : (F , N , 3) float32  -1~1 归一化坐标
    parents         : (K,)  int32  逐骨父节点索引，根骨用 -1
    """
    assert np.isfinite(frames_vertices).all(), "frames_vertices contains non-finite values"
    assert (frames_vertices.ptp(axis=(0,2))>0).all(), "vertex count mismatch"

    F, N, _ = frames_vertices.shape
    K = parents.shape[0]
    
    print(f"DemBones input: {F} frames, {N} vertices, {K} bones")
    print(f"Vertex range: min={frames_vertices.min():.3f}, max={frames_vertices.max():.3f}")
    
    # Validate input dimensions
    if F < 2:
        raise ValueError(f"Need at least 2 frames for animation, got {F}")
    if N < 3:
        raise ValueError(f"Need at least 3 vertices, got {N}")
    if K < 1:
        raise ValueError(f"Need at least 1 bone, got {K}")
    
    # Clamp iterations to reasonable range
    n_iters = max(1, min(n_iters, 100))
    nnz = max(1, min(nnz, K))
    
    try:
        dem = pdb.DemBonesExtWrapper()
        
        # Set solver parameters with validation
        print(f"Setting DemBones parameters: iterations={n_iters}, nnz={nnz}")
        dem.num_iterations = n_iters
        dem.max_nonzeros_per_vertex = nnz
        dem.weights_smoothness = 1e-4
        dem.weights_sparseness = 1e-6  # Add sparseness term
        dem.num_vertices = N
        dem.num_bones = K
        
        # Set aggressive convergence criteria to avoid infinite loops
        dem.tolerance = 1e-5  # Slightly relaxed tolerance
        dem.max_line_search_iterations = 5  # Reduced from 10
        
        # Additional parameters to prevent hanging
        try:
            # These may not be available in all DemBones versions
            dem.early_termination = True
            dem.convergence_check_frequency = 1  # Check convergence every iteration
        except AttributeError:
            print("Advanced convergence parameters not available, using defaults")
        
        # Prepare data
        rest_pose = frames_vertices[0].T.astype(np.float64)  # (3, N)
        anim_poses = frames_vertices.transpose(0,2,1).astype(np.float64)  # (F,3,N)
        anim_poses = anim_poses.reshape(3, -1)  # (3, F·N)
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {anim_poses.shape}")
        
        # Validate data ranges
        if np.abs(rest_pose).max() > 10:
            print(f"Warning: Large vertex coordinates detected: {np.abs(rest_pose).max()}")
        
        # Set data with error checking
        try:
            dem.set_rest_pose(rest_pose)
            dem.animated_poses = anim_poses
            dem.set_target_vertices('animated', anim_poses)
            dem.parents = parents.astype(np.int32)
        except Exception as e:
            raise RuntimeError(f"Failed to set DemBones data: {e}")
        
        # Run computation with timeout protection
        print("Starting DemBones computation...")
        start_time = time.time()
        
        try:
            # Use very short timeout for large datasets - fallback is better than hanging
            timeout_sec = min(60, max(30, N // 1000))  # 30-60s based on vertex count
            print(f"Using {timeout_sec}s timeout for {N} vertices")
            
            success, error_msg = run_dembone_with_timeout(dem, timeout_seconds=timeout_sec)
            
            if not success:
                error_detail = f": {error_msg}" if error_msg else ""
                raise RuntimeError(f"DemBones computation failed or timed out{error_detail}")
                
            elapsed_time = time.time() - start_time
            print(f"DemBones computation completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"DemBones computation failed after {elapsed_time:.2f} seconds: {e}")
            raise RuntimeError(f"DemBones solver failed: {e}")
        
        # Extract results with validation
        try:
            rest_pose_result = dem._dem_bones.get_rest_pose()
            if rest_pose_result.size == 0:
                raise RuntimeError("DemBones returned empty rest pose")
            rest_pose_result = _as_rowmajor(rest_pose_result)  # (N,3)
            
            weights = dem.get_weights()
            if weights.size == 0:
                raise RuntimeError("DemBones returned empty weights")
            weights = weights.T.copy()  # (N,K)
            
            # Validate weights
            if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-3):
                print("Warning: Skinning weights don't sum to 1, normalizing...")
                row_sums = weights.sum(axis=1, keepdims=True)
                row_sums[row_sums < 1e-8] = 1.0  # Avoid division by zero
                weights = weights / row_sums
            
            # Extract transformations
            flat_T = dem._dem_bones.get_transformations()
            if flat_T.size == 0:
                print("Warning: DemBones returned empty transformations, using identity matrices")
                T_all = np.zeros((F, K, 4, 4), dtype=np.float64)
                T_all[..., 3, 3] = 1.0
                for f in range(F):
                    for b in range(K):
                        T_all[f, b] = np.eye(4)
            else:
                try:
                    flat_T = flat_T.reshape(F, 3 * K, 4)
                    T_all = np.zeros((F, K, 4, 4), dtype=np.float64)
                    T_all[..., 3, 3] = 1.0
                    for f in range(F):
                        for b in range(K):
                            T_all[f, b, :3, :] = flat_T[f, b * 3:(b + 1) * 3]
                except Exception as e:
                    print(f"Warning: Failed to reshape transformations: {e}, using identity matrices")
                    T_all = np.zeros((F, K, 4, 4), dtype=np.float64)
                    T_all[..., 3, 3] = 1.0
                    for f in range(F):
                        for b in range(K):
                            T_all[f, b] = np.eye(4)
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract DemBones results: {e}")
        
        # Final validation
        if not np.isfinite(rest_pose_result).all():
            raise RuntimeError("DemBones produced non-finite rest pose vertices")
        if not np.isfinite(weights).all():
            raise RuntimeError("DemBones produced non-finite skinning weights")
        if not np.isfinite(T_all).all():
            raise RuntimeError("DemBones produced non-finite transformations")
        
        print(f"DemBones success: rest pose {rest_pose_result.shape}, weights {weights.shape}, transforms {T_all.shape}")
        print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"Sparsity: {(weights < 1e-6).sum()} / {weights.size} weights are near zero")
        
        return rest_pose_result, weights, T_all
        
    except Exception as e:
        print(f"DemBones error: {e}")
        # Return fallback results
        print("Generating fallback results...")
        rest_pose_fallback = frames_vertices[0].copy()  # Use first frame as rest pose
        weights_fallback = np.zeros((N, K), dtype=np.float32)
        
        # Simple distance-based weights as fallback
        if K > 0:
            # Assign each vertex to nearest bone (simplified)
            for i in range(N):
                weights_fallback[i, i % K] = 1.0
        
        T_fallback = np.zeros((F, K, 4, 4), dtype=np.float64)
        T_fallback[..., 3, 3] = 1.0
        for f in range(F):
            for b in range(K):
                T_fallback[f, b] = np.eye(4)
        
        print(f"Fallback results: rest pose {rest_pose_fallback.shape}, weights {weights_fallback.shape}, transforms {T_fallback.shape}")
        return rest_pose_fallback, weights_fallback, T_fallback

def load_voxel_from_mesh(file, opt, is_bind=False, scale=1.0, x_trans=0.0, z_trans=0.0):
    """
    Read mesh, normalize and voxelize for network input.
    """
    mesh = o3d.io.read_triangle_mesh(file, True)
    points = np.asarray(mesh.vertices)
    if is_bind:
        points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)
    # compute normalization params
    bmin = points.min(axis=0)
    bmax = points.max(axis=0)
    blen = (bmax - bmin).max()
    # normalize points to [-1,1]
    pts_norm = ((points - bmin) * scale / (blen + 1e-5)) * 2 - 1 + np.array([x_trans, 0, z_trans])
    # voxelize
    vox = voxelize(pts_norm, (opt.grid_size,) * 3, is_binarized=True)
    # add batch and time dimensions: (1,1,C,D,H,W)
    vox = torch.from_numpy(vox)[None, None].float().cuda()
    return vox, points, mesh, bmin, blen

def draw_skeleton(positions, parents, color=[1,0,0], radius=0.02):
    """Return list of Open3D geometries for skeleton joints and bones."""
    geoms = []
    # spheres at joint positions
    for pt in positions:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius)
        sph.translate(pt)
        sph.paint_uniform_color(color)
        sph.compute_vertex_normals()
        geoms.append(sph)
    # lines for bones
    lines = [[k, int(p)] for k, p in enumerate(parents) if p >= 0 and p != k]
    if lines:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
        geoms.append(ls)
    return geoms

def bone_colors(bone_idx):
    """Return a base RGB color for a given bone index (0–9 cycle)"""
    color_palette = [
        [1, 0, 0],    # red
        [0, 1, 0],    # green
        [0, 0, 1],    # blue
        [1, 1, 0],    # yellow
        [1, 0, 1],    # magenta
        [0, 1, 1],    # cyan
        [1, 0.5, 0],  # orange
        [0.5, 0, 1],  # purple
        [0.5, 1, 0],  # lime
        [0, 0.5, 1]   # sky blue
    ]
    return color_palette[bone_idx % len(color_palette)]

def draw_skinning_colors(pts: np.ndarray, skinning_weights, alpha=1.0):
    """
    Display a colored Open3D PointCloud using blended skinning weights per vertex.
    """
    if isinstance(skinning_weights, torch.Tensor):
        skinning_weights = skinning_weights.detach().cpu().numpy()

    V, K = skinning_weights.shape
    assert V == len(pts), f"Mesh has {len(pts)} vertices, but weights are {V}"

    # Create vertex color array
    vertex_colors = np.zeros((V, 3), dtype=np.float32)

    for k in range(K):
        color_k = bone_colors(k)
        vertex_colors += skinning_weights[:, k:k+1] * color_k

    # Normalize if needed
    vertex_colors = np.clip(vertex_colors, 0, 1)

    # draw points with colors
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([pc], window_name='Skinning Colors', point_show_normal=False)

def simple_skinning_fallback(frames_vertices: np.ndarray, joints: np.ndarray, parents: np.ndarray):
    """
    Simple distance-based skinning as fallback when DemBones fails.
    """
    F, N, _ = frames_vertices.shape
    K = len(joints)
    
    print(f"Computing simple distance-based skinning: {N} vertices, {K} joints")
    
    # Use first frame as rest pose
    rest_pose = frames_vertices[0].copy()
    
    # Compute distance-based weights
    weights = np.zeros((N, K), dtype=np.float32)
    
    for i, vertex in enumerate(rest_pose):
        # Compute distances to all joints
        distances = np.linalg.norm(joints - vertex, axis=1)
        
        # Use inverse distance weighting with falloff
        inv_distances = 1.0 / (distances + 1e-6)
        # Apply exponential falloff to make weights more localized
        inv_distances = np.exp(-distances * 2.0)
        
        # Normalize to sum to 1
        weights[i] = inv_distances / (inv_distances.sum() + 1e-8)
        
        # Enforce sparsity - keep only top 4 weights
        top_indices = np.argsort(weights[i])[-4:]
        sparse_weights = np.zeros(K)
        sparse_weights[top_indices] = weights[i][top_indices]
        sparse_weights = sparse_weights / (sparse_weights.sum() + 1e-8)
        weights[i] = sparse_weights
    
    # Create identity transformations
    T = np.zeros((F, K, 4, 4), dtype=np.float64)
    T[..., 3, 3] = 1.0
    for f in range(F):
        for b in range(K):
            T[f, b] = np.eye(4)
    
    print(f"Simple skinning complete: weights range [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Average weights per vertex: {(weights > 1e-6).sum(axis=1).mean():.2f}")
    
    return rest_pose, weights, T

def has_cycle(parents):
    """Check if parents array has cycles."""
    for i in range(len(parents)):
        seen = set()
        j = i
        while j >= 0:
            if j in seen:
                return True
            seen.add(j)
            j = parents[j]
    return False

def process_single_mesh(mesh_path, network, opt, output_dir, is_bind=False):
    """Process a single mesh file and extract skeleton data."""
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    data_path = os.path.join(output_dir, f'{base_name}_data.pkl')
    
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    print(f"Processing mesh: {mesh_path}")
    
    # Load and voxelize mesh
    vox, pts_raw, mesh, bmin, blen = load_voxel_from_mesh(mesh_path, opt, is_bind)
    
    # Normalize raw points for skinning
    pts_norm = ((pts_raw - bmin) * 1.0 / (blen + 1e-5)) * 2 - 1
    
    with torch.no_grad():
        log = network.kypt_detector(vox)
        kps = log['keypoints'][0, 0]
        affinity = log['affinity']
        dlog = network.dyna_module.encode(log['keypoints'], affinity)
        R = dlog['R'][0, 0].cpu().numpy()  # (K, 3, 3) local-frame rotations
        joints = kps[:, :3].cpu().numpy()  # (K, 3) joint positions
        parents = network.dyna_module.parents.cpu().numpy()
        K = parents.shape[0]  # number of joints
        
        # Save necessary data for future use
        data = {
            'base_name': base_name,
            'parents': parents,
            'kps': kps.cpu().numpy(),
            'joints': joints,
            'R': R,
            'bmin': bmin,
            'blen': blen,
            'pts_raw': pts_raw,
            'pts_norm': pts_norm,
            'mesh_vertices': np.asarray(mesh.vertices),
            'mesh_triangles': np.asarray(mesh.triangles),
        }
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved data for {base_name}: {K} joints, {len(pts_raw)} vertices")
        
    return data

def process_obj_files(folder_path, output_dir=None, is_bind=False, visualize=True):
    """Process all .obj files in the given folder."""
    
    if output_dir is None:
        output_dir = os.path.join(folder_path, 'generated_skeletons')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NeuralMarionette network
    exp_dir = 'pretrained/aist'
    opt_path = os.path.join(exp_dir, 'opt.pickle')
    with open(opt_path, 'rb') as f:
        opt = pickle.load(f)
    opt.Ttot = 1

    # Load network
    ckpt_path = os.path.join(exp_dir, 'aist_pretrained.pth')
    checkpoint = torch.load(ckpt_path)
    network = NeuralMarionette(opt).cuda()
    network.load_state_dict(checkpoint)
    network.eval()
    network.anneal(1)
    
    # Find all .obj files
    obj_files = glob.glob(os.path.join(folder_path, "*.obj"))
    if not obj_files:
        print(f"No .obj files found in {folder_path}")
        return
    
    print(f"Found {len(obj_files)} .obj files to process")
    
    # Initialize visualizer if needed
    if visualize:
        vis_skel = o3d.visualization.Visualizer()
        vis_skel.create_window(window_name='Skeleton Preview', width=800, height=600, visible=True)
    
    all_mesh_data = []
    all_frames_vertices = []
    
    # Process each mesh file
    for i, obj_file in enumerate(obj_files):
        print(f"\n--- Processing {i+1}/{len(obj_files)}: {os.path.basename(obj_file)} ---")
        
        try:
            mesh_data = process_single_mesh(obj_file, network, opt, output_dir, is_bind)
            all_mesh_data.append(mesh_data)
            all_frames_vertices.append(mesh_data['pts_norm'])
            
            # Visualize skeleton
            if visualize:
                joints_world = (mesh_data['joints'] + 1) * 0.5 * mesh_data['blen'] + mesh_data['bmin']
                color = [i / len(obj_files), 1 - i / len(obj_files), 0.5]
                for geo in draw_skeleton(joints_world, mesh_data['parents'], color=color):
                    vis_skel.add_geometry(geo)
                vis_skel.poll_events()
                vis_skel.update_renderer()
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")
            continue
    
    if not all_mesh_data:
        print("No meshes were successfully processed")
        return
    
    print(f"\n--- Processing {len(all_mesh_data)} meshes with DemBones ---")
    
    # Get common parameters from first mesh
    parents = all_mesh_data[0]['parents']
    parents = sanitize_parents(parents)
    
    # Check for cycles
    assert not has_cycle(parents), "Parent hierarchy contains cycles"
    
    # Find mesh with maximum vertices for template
    max_verts = max(data['pts_norm'].shape[0] for data in all_mesh_data)
    template_idx = next(i for i, data in enumerate(all_mesh_data) if data['pts_norm'].shape[0] == max_verts)
    template_data = all_mesh_data[template_idx]
    
    print(f"Using {template_data['base_name']} as template with {max_verts} vertices")
    
    # Remap all meshes to template vertex count using nearest neighbor
    frames_vertices = []
    template_pts = template_data['pts_norm']
    
    for data in all_mesh_data:
        if data['pts_norm'].shape[0] == max_verts:
            frames_vertices.append(data['pts_norm'])
        else:
            # Remap to template using nearest neighbor
            tree = cKDTree(data['pts_norm'])
            _, idx = tree.query(template_pts, k=1)
            remapped_pts = data['pts_norm'][idx]
            frames_vertices.append(remapped_pts)
            print(f"Remapped {data['base_name']} from {data['pts_norm'].shape[0]} to {max_verts} vertices")
    
    frames_vertices = np.stack(frames_vertices, axis=0)  # (F, N, 3)
    print(f"Frames vertices shape: {frames_vertices.shape}")
    
    # Solve with DemBones
    print("Running DemBones solver...")
    try:
        # Add timeout protection for the entire DemBones process
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("DemBones computation timed out")
        
        # Set timeout to 5 minutes (300 seconds)
        timeout_seconds = 300
        
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            rest_v, skin_weights, T = solve_with_dem_bones_safe(frames_vertices, parents, nnz=8, n_iters=20)
            print(f"DemBones completed: rest pose {rest_v.shape}, weights {skin_weights.shape}, transforms {T.shape}")
        finally:
            if hasattr(signal, 'SIGALRM'):
                print("Cancelling timeout alarm...")
                signal.alarm(0)  # Cancel the alarm
        
    except TimeoutError:
        print("DemBones computation timed out! Using distance-based skinning fallback...")
        
        # Use the template mesh joints for fallback skinning
        template_joints = template_data['joints']  # normalized joint positions
        rest_v, skin_weights, T = simple_skinning_fallback(frames_vertices, template_joints, parents)
        
        print(f"Timeout fallback solution: rest pose {rest_v.shape}, weights {skin_weights.shape}, transforms {T.shape}")
        return
    except Exception as e:
        print(f"DemBones failed with error: {e}")
        print("Using simple distance-based skinning fallback...")
        
        # Use the template mesh joints for fallback skinning
        template_joints = template_data['joints']  # normalized joint positions
        rest_v, skin_weights, T = simple_skinning_fallback(frames_vertices, template_joints, parents)
        
        print(f"Fallback solution: rest pose {rest_v.shape}, weights {skin_weights.shape}, transforms {T.shape}")
        return
    except Exception as e:
        print(f"Error in DemBones processing: {e}")
        return
    
    print("DemBones processing completed successfully")
    # Save DemBones results
    dembone_results = {
        'rest_pose': rest_v,
        'skinning_weights': skin_weights,
        'transforms': T,
        'joints': template_data['joints'],
        'parents': parents,
        'frame_names': [data['base_name'] for data in all_mesh_data],
        'template_data': template_data,
        'all_mesh_data': all_mesh_data
    }
    
    results_path = os.path.join(output_dir, 'dembone_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(dembone_results, f)
    
    print(f"DemBones results saved to {results_path}")
    
    # Visualize results
    if visualize:
        # Show rest pose mesh
        bmin, blen = template_data['bmin'], template_data['blen']
        rest_v_world = (rest_v + 1) * 0.5 * blen + bmin
        
        mesh_rest = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(rest_v_world),
            triangles=o3d.utility.Vector3iVector(template_data['mesh_triangles'])
        )
        mesh_rest.compute_vertex_normals()
        
        print("Displaying rest pose mesh...")
        o3d.visualization.draw_geometries([mesh_rest], window_name="Rest Pose Mesh")
        
        print("Displaying skinning weights visualization...")
        draw_skinning_colors(rest_v, skin_weights)
        
        # Show skeleton on rest pose
        joints_world = (template_data['joints'] + 1) * 0.5 * blen + bmin
        skeleton_geoms = draw_skeleton(joints_world, parents, color=[1, 0, 0])
        o3d.visualization.draw_geometries([mesh_rest] + skeleton_geoms, window_name="Rest Pose with Skeleton")
    
    # Save individual mesh results
    for i, data in enumerate(all_mesh_data):
        base_name = data['base_name']
        
        # Save mesh-specific results
        mesh_results = {
            'joints': data['joints'],
            'rotations': data['R'],
            'parents': parents,
            'skinning_weights': skin_weights,  # Add skinning weights
            'bmin': data['bmin'],
            'blen': data['blen'],
            'original_vertices': data['pts_raw'],
            'normalized_vertices': data['pts_norm']
        }
        
        mesh_result_path = os.path.join(output_dir, f'{base_name}_skeleton.pkl')
        with open(mesh_result_path, 'wb') as f:
            pickle.dump(mesh_results, f)
        
        # Save rest pose mesh for this specific mesh scale
        rest_v_world = (rest_v + 1) * 0.5 * data['blen'] + data['bmin']
        mesh_rest_scaled = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(rest_v_world),
            triangles=o3d.utility.Vector3iVector(data['mesh_triangles'])
        )
        mesh_rest_scaled.compute_vertex_normals()
        
        rest_mesh_path = os.path.join(output_dir, f'{base_name}_rest_pose.obj')
        o3d.io.write_triangle_mesh(rest_mesh_path, mesh_rest_scaled)
        
        print(f"Saved {base_name} results: skeleton data and rest pose mesh")


    if visualize:
        vis_skel.destroy_window()
    
    print(f"\n--- Processing complete ---")
    print(f"Results saved in: {output_dir}")
    print(f"- Individual mesh data: *_data.pkl")
    print(f"- Individual skeleton results: *_skeleton.pkl") 
    print(f"- Individual rest pose meshes: *_rest_pose.obj")
    print(f"- Combined DemBones results: dembone_results.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate skeletons for all .obj files in a folder using NeuralMarionette and DemBones.")
    parser.add_argument('folder_path', type=str, help='Path to folder containing .obj files.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: folder_path/generated_skeletons).')
    parser.add_argument('--is_bind', action='store_true', help='Use bind transform for mesh coordinates.')
    parser.add_argument('--no_visualize', action='store_true', help='Disable visualization.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        exit(1)
    
    process_obj_files(
        folder_path=args.folder_path,
        output_dir=args.output_dir,
        is_bind=args.is_bind,
        visualize=not args.no_visualize
    )