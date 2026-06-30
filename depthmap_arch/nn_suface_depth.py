import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MultiClassUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        return self.out_conv(x)


# ==========================================
# 2. MOUSE INTERACTION & STATE
# ==========================================
mouse_state = {
    'is_down': False, 
    'x': 0, 
    'y': 0,
    'spawn_z': 2.0  # Default spawn depth (2 meters)
}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: 
        mouse_state['is_down'] = True
    elif event == cv2.EVENT_LBUTTONUP: 
        mouse_state['is_down'] = False
    elif event == cv2.EVENT_MOUSEMOVE: 
        mouse_state['x'], mouse_state['y'] = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Adjust spawn depth using the scroll wheel
        if flags > 0:
            mouse_state['spawn_z'] += 0.5
        else:
            mouse_state['spawn_z'] -= 0.5
        
        # Clamp values so we don't spawn behind the camera or too far away
        mouse_state['spawn_z'] = max(0.5, min(mouse_state['spawn_z'], 15.0))


# ==========================================
# 3. FULL AR PHYSICS ENGINE (NN + RICOCHET)
# ==========================================
def run_multi_class_ar_physics(image_path, model_path="multi_class_room_epoch_10.pth"):
    print("==================================================")
    print("STARTING: FULL ROOM AR RICOCHET ENGINE (AI DRIVEN)")
    print("==================================================")

    # A. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiClassUNet(in_channels=1, out_channels=4).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("[1/4] ✅ Multi-Class U-Net Loaded!")
    except Exception as e:
        print(f"❌ Error loading '{model_path}'. Check path.\n{e}")
        input("\nPress Enter to exit...")
        return

    # B. Load Image & Scale it for Laptop Screens
    depth_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        print(f"❌ Error loading '{image_path}'.")
        input("\nPress Enter to exit...")
        return
        
    h, w = depth_img.shape
    
    # Automatically scale down large images so the window fits on screen
    max_width = 800
    if w > max_width:
        scale = max_width / w
        depth_img = cv2.resize(depth_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = depth_img.shape
        
    fx, fy = max(h, w), max(h, w)
    cx, cy = w / 2.0, h / 2.0

    # C. AI Semantic Prediction
    print("[2/4] Analyzing full room structure...")
    input_img = cv2.resize(depth_img, (256, 256), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        class_predictions = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
    semantic_mask = cv2.resize(class_predictions.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # D. Build Mathematical Collision Planes
    print("[3/4] Converting AI masks into rigid body physics planes...")
    Z = 4.0 * (1.0 - (depth_img.astype(float) / 255.0)) + 0.01
    u_indices, v_indices = np.meshgrid(np.arange(w), np.arange(h))
    X = (u_indices - cx) * Z / fx
    Y = (v_indices - cy) * Z / fy

    points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    semantic_flat = semantic_mask.flatten()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    planes_data = []
    
    # Iterate through Floors(1), Side Walls(2), and Front Walls(3)
    for class_id in [1, 2, 3]:
        class_indices = np.where(semantic_flat == class_id)[0]
        if len(class_indices) < 200: continue
            
        class_pcd = pcd.select_by_index(class_indices)
        
        # 🚀 SPEED & STABILITY FIX: Downsample before DBSCAN to prevent OOM crash
        class_pcd = class_pcd.voxel_down_sample(voxel_size=0.04)
        if len(class_pcd.points) < 30: continue
        
        # DBSCAN to separate multiple walls/tables
        labels = np.array(class_pcd.cluster_dbscan(eps=0.2, min_points=15, print_progress=False))
        if len(labels) == 0 or labels.max() == -1: continue
            
        for label in np.unique(labels):
            if label == -1: continue
            cluster_indices = np.where(labels == label)[0]
            
            # Adjusted threshold since we downsampled
            if len(cluster_indices) < 30: continue
            
            cluster_pcd_obj = class_pcd.select_by_index(cluster_indices)
            
            try:
                # Mathematical Plane Equation (ax + by + cz + d = 0)
                plane_model, _ = cluster_pcd_obj.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
                a, b, c, d = plane_model
                
                plane_normal = np.array([a, b, c])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                
                # 🚀 CRASH/INTERACTION FIX: Orient the Normal towards the camera!
                # If a normal points away from the camera, the ball won't bounce properly off the front of it.
                centroid = np.mean(np.asarray(cluster_pcd_obj.points), axis=0)
                if np.dot(plane_normal, centroid) > 0:
                    plane_normal = -plane_normal
                    d = -d
                
                # Boundary calculation (So balls can roll off edges)
                pts_3d = np.asarray(cluster_pcd_obj.points)
                u_pts = (pts_3d[:, 0] * fx / pts_3d[:, 2]) + cx
                v_pts = (pts_3d[:, 1] * fy / pts_3d[:, 2]) + cy
                
                pts_2d = np.vstack((u_pts, v_pts)).T.astype(np.int32)
                
                if len(pts_2d) < 3: continue # Need at least 3 points for a hull
                
                hull = cv2.convexHull(pts_2d)
                
                boundary_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(boundary_mask, [hull], 255)
                
                planes_data.append({
                    'normal': plane_normal,
                    'd': d,
                    'mask': boundary_mask,
                    'class_id': class_id
                })
            except Exception as e:
                print(f"⚠️ Skipped a problematic physics plane: {e}")
                continue

    if len(planes_data) == 0:
        print("❌ Warning: Could not generate any valid colliders. The AI might not have found clear planes.")
        input("\nPress Enter to exit...")
        return

    print(f"      ✅ Loaded {len(planes_data)} distinct physics barriers!")

    # E. AR Loop
    print("[4/4] Launching Interactive AR Game.")
    print("      - CLICK AND DRAG to throw balls!")
    print("      - SCROLL WHEEL (or W/S keys) to change spawn depth!")
    
    window_name = "AR Ricochet Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)
    cv2.setMouseCallback(window_name, mouse_callback)

    bg_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
    spheres = []
    dt = 0.03 
    gravity = np.array([0.0, 9.8, 0.0]) # Standard gravity

    while True:
        frame = bg_img.copy()
        
        # --- UI Overlay ---
        current_z = mouse_state['spawn_z']
        cv2.putText(frame, f"Spawn Depth (Z): {current_z:.1f}m", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "[Scroll Wheel / W / S] to change", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw a 3D aiming crosshair at the mouse position
        mx, my = mouse_state['x'], mouse_state['y']
        crosshair_size = max(5, int(fx * 0.05 / current_z)) # Crosshair scales down with distance
        cv2.drawMarker(frame, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, crosshair_size, 2)
        
        # --- Spawn Logic ---
        if mouse_state['is_down']:
            # 🚀 SPAWN FIX: Use the actual scroll wheel depth, not a hardcoded 0.5!
            spawn_z = current_z 
            spawn_x = (mx - cx) * spawn_z / fx
            spawn_y = (my - cy) * spawn_z / fy
            
            spheres.append({
                'pos': np.array([spawn_x, spawn_y, spawn_z]),
                'vel': np.array([
                    np.random.uniform(-3.0, 3.0), # Higher X scatter so they hit side walls
                    np.random.uniform(-3.0, 1.0), # Throw slightly upwards
                    np.random.uniform(3.0, 8.0)   # Z forward throw force
                ]),
                'radius': 0.04, 
                'color': (np.random.randint(50, 255), np.random.randint(50, 255), 255)
            })

        # --- Physics Update ---
        alive_spheres = []
        for s in spheres:
            s['vel'] += gravity * dt
            s['pos'] += s['vel'] * dt
            s['vel'] *= 0.99 # Air drag
            
            pos = s['pos']
            if pos[2] <= 0.1 or pos[2] > 20.0 or pos[1] > 10.0:
                continue 
                
            u = int((pos[0] * fx / pos[2]) + cx)
            v = int((pos[1] * fy / pos[2]) + cy)
            
            # Full Room Multi-Plane Collision
            has_bounced = False
            for plane in planes_data:
                if has_bounced: break 
                
                dist_to_plane = np.dot(plane['normal'], pos) + plane['d']
                
                is_over_plane = False
                if 0 <= u < w and 0 <= v < h:
                    if plane['mask'][v, u] == 255:
                        is_over_plane = True

                # Bounce Logic
                # 🚀 TELEPORTATION FIX: Ensure the ball isn't already miles behind the wall before snapping it forward!
                if -0.2 < dist_to_plane < s['radius'] and is_over_plane:
                    s['pos'] += plane['normal'] * (s['radius'] - dist_to_plane)
                    
                    v_dot_n = np.dot(s['vel'], plane['normal'])
                    if v_dot_n < 0:
                        # Make walls (class 2 & 3) bounce slightly harder than floors (class 1)
                        restitution = 0.85 if plane['class_id'] in [2, 3] else 0.65
                        s['vel'] = s['vel'] - (1.0 + restitution) * v_dot_n * plane['normal']
                    
                    has_bounced = True
            
            alive_spheres.append(s)
        spheres = alive_spheres

        # --- Render ---
        spheres.sort(key=lambda x: x['pos'][2], reverse=True)

        for s in spheres:
            pos = s['pos']
            u = int((pos[0] * fx / pos[2]) + cx)
            v = int((pos[1] * fy / pos[2]) + cy)
            r_2d = int((fx * s['radius']) / pos[2])
            
            if r_2d > 0:
                cv2.circle(frame, (u, v), r_2d, s['color'], -1)
                hl_offset = max(1, int(r_2d * 0.3))
                hl_rad = max(1, int(r_2d * 0.2))
                cv2.circle(frame, (u - hl_offset, v - hl_offset), hl_rad, (255, 255, 255), -1)
                cv2.circle(frame, (u, v), r_2d, (0, 0, 0), 1)

        cv2.imshow(window_name, frame)
        
        # --- Keyboard Inputs ---
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'): # Keyboard fallback: Increase Z
            mouse_state['spawn_z'] = min(15.0, mouse_state['spawn_z'] + 0.5)
        elif key == ord('s'): # Keyboard fallback: Decrease Z
            mouse_state['spawn_z'] = max(0.5, mouse_state['spawn_z'] - 0.5)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Point this to your depth map and your trained weights file
    run_multi_class_ar_physics("depthmap6.png", "multi_class_room_epoch_10.pth")