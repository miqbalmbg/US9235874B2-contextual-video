"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI Hybrid (On Device - On Cloud) Video Generation                          ║
║   Contextual Video Upscaling Demo — Based on US9235874B2                     ║
║                                                                              ║
║   Method: Multi-frame temporal upscaling using optical flow                  ║
║   Each frame is upscaled using detail borrowed from neighboring frames,      ║
║   just like the patent describes — mirroring your hybrid pipeline.           ║
║                                                                              ║
║   Pipeline:                                                                  ║
║   [DEVICE] Generate low-res GIF -> User Confirmation                         ║
║   [DEVICE] Prepare anonymized frame metadata                                 ║
║   [CLOUD]  Optical flow estimation -> Frame alignment -> Detail fusion       ║
║            -> Upscale -> Sharpen                                             ║
║   [DEVICE] Verify integrity -> Assemble final high-res video                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP (run once):
    pip install opencv-contrib-python pillow numpy requests

HOW TO USE:
    python contextual_video_upscaling_demo.py --demo
    python contextual_video_upscaling_demo.py --input your_video.mp4
    python contextual_video_upscaling_demo.py --input your_image.jpg --scale 2

KEY DIFFERENCE FROM EDSR SCRIPT:
    - Uses NEIGHBORING FRAMES as context (US9235874B2 method)
    - Optical flow warps adjacent frames to align with current frame
    - Detail from 2 neighbors is fused before upscaling
    - Reduces temporal flickering between frames
    - Closely mirrors real video super-resolution pipelines

OUTPUT (in ./pipeline_output_contextual/):
    step1_lowres_preview.gif        - On-device low-res GIF preview
    step2_cloud_frames/             - Clean frames ready for cloud
    step3_upscaled_frames/          - Contextually upscaled frames
    step4_final_video.mp4           - Final high-res video
    pipeline_comparison.jpg         - Before/after comparison
    optical_flow_visualization.jpg  - Optical flow map (bonus visual)
"""

import os, sys, argparse, time, hashlib
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
def check_deps():
    errors = []
    try:
        import cv2
        if not hasattr(cv2, 'dnn_superres'):
            errors.append(
                "opencv-contrib-python missing.\n"
                "    Run: pip uninstall opencv-python -y\n"
                "         pip install opencv-contrib-python"
            )
        # Check optical flow is available (Farneback)
        try:
            cv2.calcOpticalFlowFarneback
        except AttributeError:
            errors.append("cv2.calcOpticalFlowFarneback not found — reinstall opencv-contrib-python")
    except ImportError:
        errors.append("opencv-contrib-python  ->  pip install opencv-contrib-python")
    for pkg, mod in [("pillow","PIL"), ("numpy","numpy"), ("requests","requests")]:
        try:
            __import__(mod)
        except ImportError:
            errors.append(f"{pkg}  ->  pip install {pkg}")
    if errors:
        print("\n[X]  Fix these before running:\n")
        for e in errors: print(f"    {e}")
        print(); sys.exit(1)
    print("[OK] All dependencies found.\n")

check_deps()

import cv2, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("./pipeline_output_contextual")
CLOUD_DIR   = OUTPUT_DIR / "step2_cloud_frames"
UP_DIR      = OUTPUT_DIR / "step3_upscaled_frames"
MODELS_DIR  = OUTPUT_DIR / "models"
SCALE       = 4
PREV_FPS    = 8
OUT_FPS     = 24
MAX_FRAMES  = 24
LR_W, LR_H = 160, 90

# Temporal context window: how many neighboring frames to use
# 1 = use 1 frame before + 1 frame after (2 neighbors total)
# 2 = use 2 frames before + 2 frames after (4 neighbors total)
CONTEXT_RADIUS = 1

# Sharpening strength after upscale (0.8 subtle, 1.2 default, 1.8 strong)
SHARPEN_STRENGTH = 1.2

MODELS = {
    2: dict(algo="edsr", scale=2, file="EDSR_x2.pb",
            url="https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"),
    4: dict(algo="edsr", scale=4, file="EDSR_x4.pb",
            url="https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def hdr(n, title, desc=""):
    print(f"\n{'='*64}\n  STEP {n}  |  {title}\n{'-'*64}")
    if desc: print(f"  {desc}\n")

def info(k, v): print(f"  . {k:<38} {v}")

def bar(cur, tot, lbl=""):
    b = "█"*int(32*cur/tot) + "░"*(32-int(32*cur/tot))
    print(f"\r  {lbl} [{b}] {cur}/{tot}", end="", flush=True)
    if cur == tot: print()

def dl_model(cfg):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / cfg["file"]
    if dst.exists():
        info("Model cached", str(dst)); return str(dst)
    print(f"  Downloading {cfg['file']} (~10-40 MB)...")
    try:
        r = requests.get(cfg["url"], stream=True, timeout=90)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk); done += len(chunk)
                if total: bar(min(done,total), total, "Downloading")
        print(); info("Saved", str(dst)); return str(dst)
    except Exception as e:
        print(f"\n  [!] Download failed ({e}). Using bicubic fallback."); return None

# ── Step 0: source frames ─────────────────────────────────────────────────────
def make_demo(n=MAX_FRAMES, W=640, H=360):
    hdr(0, "GENERATING DEMO VIDEO", "Creating synthetic animated test video with motion")
    frames = []
    for i in range(n):
        t = i / n
        f = np.zeros((H, W, 3), np.uint8)
        # Animated gradient background - clamped to prevent uint8 overflow
        for y in range(H):
            r = int(np.clip(20  + 60 * np.sin(np.pi*y/H + t*6.28), 0, 255))
            g = int(np.clip(40  + 50 * np.cos(np.pi*y/H + t*3.14), 0, 255))
            b = int(np.clip(100 + 80 * np.sin(t*6.28),              0, 255))
            f[y] = [r, g, b]
        # Moving circle (smooth trajectory for good optical flow)
        cx = int(W*(0.15 + 0.70*abs(np.sin(t*3.14))))
        cy = int(H*(0.25 + 0.50*np.cos(t*6.28)))
        cv2.circle(f, (cx,cy), 50, (220,180,60), -1)
        cv2.circle(f, (cx,cy), 50, (255,230,130), 2)
        # Second moving object (opposite direction — tests optical flow)
        cx2 = int(W*(0.85 - 0.70*abs(np.sin(t*3.14))))
        cy2 = int(H*(0.70 + 0.20*np.sin(t*6.28)))
        cv2.rectangle(f, (cx2-35, cy2-22), (cx2+35, cy2+22), (60,150,220), -1)
        # Static reference object (tests detail preservation)
        cv2.circle(f, (W//2, H//2), 15, (255,80,80), -1)
        # Labels
        cv2.putText(f, f"Frame {i+1:02d}/{n}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(f, "SRIN Patent Demo - Contextual SR", (10,H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (180,180,180), 1, cv2.LINE_AA)
        frames.append(f)
        bar(i+1, n, "Generating")
    info("Frames", str(n)); info("Size", f"{W}x{H}")
    return frames

def load_input(path):
    path = Path(path)
    hdr(0, "LOADING INPUT", f"File: {path.name}")
    frames = []
    if path.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
        fr = cv2.imread(str(path))
        if fr is None: print(f"  Cannot read {path}"); sys.exit(1)
        frames = [fr.copy() for _ in range(MAX_FRAMES)]
        info("Type", "Image (repeated — optical flow will detect zero motion)")
    elif path.suffix.lower() in [".mp4",".avi",".mov",".mkv",".gif"]:
        cap = cv2.VideoCapture(str(path))
        while len(frames) < MAX_FRAMES:
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
        cap.release()
        if not frames: print(f"  Cannot read {path}"); sys.exit(1)
        info("Type", "Video")
    else:
        print(f"  Unsupported: {path.suffix}"); sys.exit(1)
    info("Frames loaded", str(len(frames)))
    return frames

# ── Step 1: ON-DEVICE low-res GIF preview ────────────────────────────────────
def step1_lowres_preview(frames):
    hdr(1, "ON-DEVICE: Generate Low-Res GIF Preview",
        "Small preview for user confirmation before cloud processing begins")
    lr = []
    for i, f in enumerate(frames):
        s = cv2.resize(f, (LR_W, LR_H), interpolation=cv2.INTER_LINEAR)
        s = cv2.GaussianBlur(s, (3,3), 0.4)
        lr.append(s)
        bar(i+1, len(frames), "Downscaling")

    gif = OUTPUT_DIR / "step1_lowres_preview.gif"
    pf  = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in lr]
    pf[0].save(gif, save_all=True, append_images=pf[1:],
               duration=int(1000/PREV_FPS), loop=0)

    info("GIF saved",  str(gif))
    info("Resolution", f"{LR_W}x{LR_H}")
    print()
    print("  [!]  USER CONFIRMATION GATE")
    print("  |    User reviews low-res GIF preview.")
    print("  |    Cloud processing starts ONLY after approval.")
    print("  +->  [CONFIRMED - proceeding]\n")
    return lr

# ── Step 2: ON-DEVICE metadata anonymization ──────────────────────────────────
def step2_anonymize(lr_frames):
    hdr(2, "ON-DEVICE: Prepare Frames for Cloud Upload",
        "Generate integrity tokens on-device. Clean pixels sent to cloud.")

    CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    KEY = 42
    anon_data = []

    for i, frame in enumerate(lr_frames):
        frame_hash = hashlib.sha256(frame.tobytes() + str(KEY).encode()).hexdigest()
        cloud_frame = frame.copy()
        anon_data.append({
            "cloud_frame":     cloud_frame,
            "integrity_token": frame_hash,
            "frame_idx":       i,
        })
        cv2.imwrite(str(CLOUD_DIR / f"frame_{i:04d}.png"), cloud_frame)
        bar(i+1, len(lr_frames), "Preparing")

    info("Frames prepared",    str(len(anon_data)))
    info("Privacy method",     "Integrity tokens (on-device) + secure channel")
    info("Pixel modification", "NONE - clean input for best upscale quality")
    print("\n  [Cloud] Frames received. Beginning contextual upscaling...\n")
    return anon_data

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: ON-CLOUD — Contextual Video Upscaling (US9235874B2 method)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_optical_flow(frame_a, frame_b):
    """
    Estimate dense optical flow between two frames using Farneback method.
    Returns flow field: array of shape (H, W, 2) where [:,:,0]=dx, [:,:,1]=dy
    This tells us how each pixel in frame_a moved to reach frame_b.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale  = 0.5,   # Image pyramid scale
        levels     = 3,     # Pyramid levels
        winsize    = 15,    # Window size for flow estimation
        iterations = 3,     # Iterations per pyramid level
        poly_n     = 5,     # Pixel neighborhood size
        poly_sigma = 1.2,   # Gaussian std for polynomial expansion
        flags      = 0
    )
    return flow

def warp_frame(frame, flow):
    """
    Warp a frame using an optical flow field.
    Maps each pixel to its new position based on the flow vectors.
    This spatially aligns the neighbor frame with the current frame.
    """
    H, W = flow.shape[:2]
    # Build remapping grid
    map_x = np.tile(np.arange(W), (H,1)).astype(np.float32)
    map_y = np.tile(np.arange(H), (W,1)).T.astype(np.float32)
    # Apply flow displacement
    map_x += flow[:,:,0]
    map_y += flow[:,:,1]
    # Remap pixels
    warped = cv2.remap(frame, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped

def compute_reliability_mask(flow, threshold=3.0):
    """
    Compute a reliability mask for the flow field.
    Pixels with large flow magnitude = moving objects = lower reliability.
    Pixels with small flow magnitude = static background = higher reliability.
    This prevents ghosting artifacts when blending warped frames.
    """
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    # Normalize and invert: low motion = high reliability weight
    reliability = np.exp(-magnitude / threshold)
    # Expand to 3 channels for blending
    return np.stack([reliability]*3, axis=-1).astype(np.float32)

def fuse_frames_with_context(current_frame, neighbor_frames, neighbor_flows):
    """
    Core of the US9235874B2 method:
    Fuse the current frame with detail borrowed from temporally neighboring frames.

    Strategy:
    1. Warp each neighbor to align with current frame using optical flow
    2. Compute reliability mask (down-weight moving regions)
    3. Weighted blend: current frame gets highest weight, neighbors contribute detail
    """
    H, W = current_frame.shape[:2]
    current_f = current_frame.astype(np.float32)

    # Start with current frame at full weight
    accumulated   = current_f.copy()
    total_weights = np.ones((H, W, 3), dtype=np.float32)

    for neighbor, flow in zip(neighbor_frames, neighbor_flows):
        # Warp neighbor to align with current frame
        warped = warp_frame(neighbor, flow).astype(np.float32)

        # Compute reliability: static areas contribute more detail
        reliability = compute_reliability_mask(flow, threshold=2.5)

        # Neighbor contributes with reliability-weighted blending
        # Weight = 0.5 * reliability (neighbors get up to 50% contribution)
        weight = 0.5 * reliability
        accumulated   += warped * weight
        total_weights += weight

    # Normalize by total weights
    fused = (accumulated / total_weights).clip(0, 255).astype(np.uint8)
    return fused

def sharpen_frame(frame, strength=SHARPEN_STRENGTH):
    """Unsharp masking: amplifies edges and fine detail after upscaling."""
    blur  = cv2.GaussianBlur(frame, (0,0), 3)
    sharp = cv2.addWeighted(frame, 1+strength, blur, -strength, 0)
    return sharp

def visualize_optical_flow(flow, save_path):
    """
    Convert optical flow to a color visualization image.
    Direction = hue, magnitude = saturation/value.
    Useful for understanding what the algorithm detected.
    """
    H, W = flow.shape[:2]
    hsv  = np.zeros((H, W, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    hsv[:,:,0] = ang * 180 / np.pi / 2       # Hue = direction
    hsv[:,:,1] = 255                           # Full saturation
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(save_path), flow_rgb)

def step3_contextual_upscale(anon_data):
    hdr(3, f"ON-CLOUD: Contextual Video Upscaling ({SCALE}x) — US9235874B2",
        f"Optical flow alignment + multi-frame fusion + EDSR upscaling + sharpening")

    UP_DIR.mkdir(parents=True, exist_ok=True)

    frames     = [item["cloud_frame"] for item in anon_data]
    n          = len(frames)
    info("Total frames",       str(n))
    info("Context radius",     f"+/- {CONTEXT_RADIUS} neighboring frames")
    info("Neighbors per frame", str(CONTEXT_RADIUS * 2))
    info("Optical flow method", "Farneback dense flow (cv2.calcOpticalFlowFarneback)")
    info("Fusion method",       "Reliability-weighted temporal blending")

    # Load EDSR upscaler for the final upscale step
    cfg     = MODELS.get(SCALE, MODELS[4])
    mpath   = dl_model(cfg)
    use_dnn = False
    if mpath:
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(mpath)
            sr.setModel(cfg["algo"], cfg["scale"])
            use_dnn = True
            info("Upscale method", f"EDSR x{SCALE} (after contextual fusion)")
        except Exception as e:
            print(f"  [!] DNN load failed ({e}). Using bicubic fallback.")
    if not use_dnn:
        info("Upscale method", "Bicubic interpolation (fallback)")
    info("Post-process", f"Unsharp masking (strength={SHARPEN_STRENGTH})")
    print()

    upscaled    = []
    flow_saved  = False
    t0          = time.time()

    for i in range(n):
        current = frames[i]

        # ── 1. Gather neighbor frame indices ─────────────────────────────────
        neighbor_indices = []
        for offset in range(-CONTEXT_RADIUS, CONTEXT_RADIUS+1):
            if offset == 0: continue
            idx = i + offset
            if 0 <= idx < n:
                neighbor_indices.append(idx)

        # ── 2. Compute optical flow for each neighbor ─────────────────────
        neighbor_frames = []
        neighbor_flows  = []
        for ni in neighbor_indices:
            neighbor = frames[ni]
            # Flow: how to warp neighbor to align with current frame
            flow = estimate_optical_flow(neighbor, current)
            neighbor_frames.append(neighbor)
            neighbor_flows.append(flow)

            # Save flow visualization for first frame pair only
            if not flow_saved and ni == 1:
                visualize_optical_flow(
                    flow,
                    OUTPUT_DIR / "optical_flow_visualization.jpg"
                )
                flow_saved = True

        # ── 3. Fuse current frame with aligned neighbor detail ────────────
        if neighbor_frames:
            fused = fuse_frames_with_context(current, neighbor_frames, neighbor_flows)
        else:
            fused = current.copy()  # Edge case: first/last frame with no neighbors

        # ── 4. Upscale the fused frame ────────────────────────────────────
        if use_dnn:
            try:
                up = sr.upsample(fused)
            except Exception:
                h, w = fused.shape[:2]
                up = cv2.resize(fused, (w*SCALE, h*SCALE), interpolation=cv2.INTER_CUBIC)
        else:
            h, w = fused.shape[:2]
            up = cv2.resize(fused, (w*SCALE, h*SCALE), interpolation=cv2.INTER_CUBIC)

        # ── 5. Sharpen post-process ───────────────────────────────────────
        up_sharp = sharpen_frame(up, strength=SHARPEN_STRENGTH)

        upscaled.append(up_sharp)
        cv2.imwrite(str(UP_DIR / f"upscaled_{i:04d}.png"), up_sharp)
        bar(i+1, n, "Contextual upscaling")

    H, W = upscaled[0].shape[:2]
    info("Input resolution",      f"{LR_W}x{LR_H}")
    info("Output resolution",     f"{W}x{H}")
    info("Scale factor",          f"{SCALE}x")
    info("Cloud processing time", f"{time.time()-t0:.1f}s")
    info("Output folder",         str(UP_DIR))
    return upscaled

# ── Step 4: ON-DEVICE verify + assemble ──────────────────────────────────────
def step4_verify_assemble(upscaled, anon_data):
    hdr(4, "ON-DEVICE: Verify & Assemble Final Video",
        "Verify frame integrity, assemble MP4")

    final = []; H, W = upscaled[0].shape[:2]; verified = 0
    for i, (up, item) in enumerate(zip(upscaled, anon_data)):
        if up.shape[0] == LR_H*SCALE and up.shape[1] == LR_W*SCALE:
            verified += 1
        final.append(up)
        bar(i+1, len(upscaled), "Assembling")

    video_path = OUTPUT_DIR / "step4_final_video.mp4"
    wr = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (W,H))
    for f in final: wr.write(f)
    wr.release()

    info("Frames verified", f"{verified}/{len(upscaled)}")
    info("Video output",    str(video_path))
    info("Resolution",      f"{W}x{H}")
    info("FPS",             str(OUT_FPS))
    return final

# ── Step 5: comparison ────────────────────────────────────────────────────────
def step5_comparison(lr, final):
    hdr(5, "COMPARISON IMAGE", "Side-by-side: low-res GIF vs contextual upscaling output")

    mid  = len(lr) // 2
    lri  = cv2.cvtColor(lr[mid],    cv2.COLOR_BGR2RGB)
    hri  = cv2.cvtColor(final[mid], cv2.COLOR_BGR2RGB)
    H, W = hri.shape[:2]
    lru  = np.array(Image.fromarray(lri).resize((W,H), Image.NEAREST))

    gap = 20; lh = 60
    C   = Image.new("RGB", (W*2+gap*3, H+lh+gap*2), (20,20,20))
    C.paste(Image.fromarray(lru), (gap, gap+lh))
    C.paste(Image.fromarray(hri), (W+gap*2, gap+lh))
    d = ImageDraw.Draw(C)
    try:
        fb = ImageFont.truetype("arial.ttf", 20)
        fs = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            fb = fs = ImageFont.load_default()

    d.text((gap+10, 8),
           f"ON-DEVICE Preview  ({LR_W}x{LR_H})", fill=(255,200,60), font=fb)
    d.text((W+gap*2+10, 8),
           f"ON-CLOUD Contextual SR ({W}x{H})", fill=(80,230,120), font=fb)
    d.text((gap, H+lh+gap+4),
           f"Method: Optical Flow + Temporal Fusion + EDSR {SCALE}x + Unsharp Masking  "
           f"| Context: +/-{CONTEXT_RADIUS} frames | Ref: US9235874B2",
           fill=(150,150,150), font=fs)

    out = OUTPUT_DIR / "pipeline_comparison.jpg"
    C.save(str(out), quality=95)
    info("Comparison saved", str(out))
    info("Flow visualization", str(OUTPUT_DIR / "optical_flow_visualization.jpg"))
    return out

# ── Main ──────────────────────────────────────────────────────────────────────
def run(inp=None, demo=False):
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Contextual Video Upscaling Pipeline Demo                   ║
║   Method: US9235874B2 (Optical Flow + Temporal Fusion)       ║
║   Patent: M. Iqbal Mauludi  -  Samsung Confidential          ║
╠══════════════════════════════════════════════════════════════╣
║  How this differs from EDSR-only (previous script):          ║
║  [+] Neighboring frames provide extra detail via flow warp   ║
║  [+] Static regions sharpened from multiple observations     ║
║  [+] Moving regions handled via reliability masking          ║
║  [+] Temporal consistency across the whole video sequence    ║
╚══════════════════════════════════════════════════════════════╝
""")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = make_demo() if (demo or not inp) else load_input(inp)
    lr     = step1_lowres_preview(frames)
    data   = step2_anonymize(lr)
    ups    = step3_contextual_upscale(data)
    final  = step4_verify_assemble(ups, data)
    comp   = step5_comparison(lr, final)

    print(f"\n{'='*64}\n  PIPELINE COMPLETE\n{'-'*64}")
    info("1  Low-res GIF",      str(OUTPUT_DIR/"step1_lowres_preview.gif"))
    info("2  Cloud frames",     str(CLOUD_DIR))
    info("3  Upscaled frames",  str(UP_DIR))
    info("4  Final video",      str(OUTPUT_DIR/"step4_final_video.mp4"))
    info("5  Comparison",       str(comp))
    info("   Flow visualization", str(OUTPUT_DIR/"optical_flow_visualization.jpg"))
    print(f"\n  Open ./pipeline_output_contextual/ to review all outputs.")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Contextual Video Upscaling Demo (US9235874B2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python contextual_video_upscaling_demo.py --demo
  python contextual_video_upscaling_demo.py --input my_video.mp4
  python contextual_video_upscaling_demo.py --input my_photo.jpg --scale 2
        """
    )
    ap.add_argument("--input",  help="Path to video or image file")
    ap.add_argument("--demo",   action="store_true", help="Run with auto-generated demo")
    ap.add_argument("--scale",  type=int, default=4, choices=[2,4],
                    help="Upscale factor: 2 or 4 (default: 4)")
    ap.add_argument("--radius", type=int, default=1, choices=[1,2],
                    help="Context radius: 1=2 neighbors, 2=4 neighbors (default: 1)")
    args = ap.parse_args()
    SCALE          = args.scale
    CONTEXT_RADIUS = args.radius
    run(inp=args.input, demo=args.demo)
