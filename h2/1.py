import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_fig(path, dpi=200):
    ensure_dir(os.path.dirname(path) or ".")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {path}")


def normalize_to_uint8(x: np.ndarray, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn + eps)
    return (y * 255.0).clip(0, 255).astype(np.uint8)


def read_gray_fixed():
    path = r"F:\h2\test.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"读图失败：{path}")
    return img.astype(np.float32)


def read_bgr_fixed():
    path = r"F:\h2\test.png"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读图失败：{path}")
    return img


# 手写卷积（2D）
def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kH, kW = kernel.shape
    if (kH % 2 == 0) or (kW % 2 == 0) or (kH < 3) or (kW < 3):
        raise ValueError("kernel 必须是 >=3 的奇数尺寸")

    pad_h, pad_w = kH // 2, kW // 2
    padded = cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w,
        borderType=cv2.BORDER_REFLECT_101
    )

    out = np.zeros_like(img, dtype=np.float32)
    k = kernel[::-1, ::-1].astype(np.float32)

    H, W = img.shape
    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kH, j:j + kW]
            out[i, j] = float(np.sum(patch * k))
    return out


#  1) Sobel 
def sobel_gradients(gray: np.ndarray):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    gx = conv2d(gray, Kx)
    gy = conv2d(gray, Ky)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang


#  2) Canny 
def non_max_suppression(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    H, W = mag.shape
    out = np.zeros((H, W), dtype=np.float32)

    angle = ang * 180.0 / np.pi
    angle[angle < 0] += 180.0

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i - 1, j + 1]
                r = mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                q = mag[i - 1, j]
                r = mag[i + 1, j]
            else:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            out[i, j] = mag[i, j] if (mag[i, j] >= q and mag[i, j] >= r) else 0.0

    return out


def double_threshold(img: np.ndarray, low_ratio=0.08, high_ratio=0.20):
    mx = float(img.max())
    high = mx * float(high_ratio)
    low = mx * float(low_ratio)

    strong = 255
    weak = 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong, low, high


def hysteresis(thres: np.ndarray, weak=75, strong=255) -> np.ndarray:
    H, W = thres.shape
    out = thres.copy()

    stack = list(zip(*np.where(out == strong)))
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and out[ni, nj] == weak:
                    out[ni, nj] = strong
                    stack.append((ni, nj))

    out[out != strong] = 0
    return out


def canny_manual(gray: np.ndarray,
                 blur_ksize=5, blur_sigma=1.2,
                 low_ratio=0.08, high_ratio=0.20):
    if blur_ksize % 2 == 0 or blur_ksize < 3:
        raise ValueError("blur_ksize 必须是 >=3 的奇数")

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    gx, gy, mag, ang = sobel_gradients(blurred)
    nms = non_max_suppression(mag, ang)
    thres, weak, strong, low, high = double_threshold(nms, low_ratio=low_ratio, high_ratio=high_ratio)
    edges = hysteresis(thres, weak=weak, strong=strong)

    return {
        "blurred": blurred,
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "nms": nms,
        "thres": thres,
        "edges": edges,
        "low": low,
        "high": high
    }


#  3) Harris 
def harris_manual(gray: np.ndarray, k=0.04, win_sigma=1.5, thresh_ratio=0.01, nms_ksize=3):
    gx, gy, _, _ = sobel_gradients(gray)

    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy

    Sxx = cv2.GaussianBlur(Ixx, (0, 0), win_sigma)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), win_sigma)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), win_sigma)

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - float(k) * (trace ** 2)

    R_max = float(R.max())
    thresh = float(thresh_ratio) * R_max
    R_th = np.zeros_like(R, dtype=np.float32)
    mask = R > thresh
    R_th[mask] = R[mask]

    if nms_ksize % 2 == 0 or nms_ksize < 3:
        raise ValueError("nms_ksize 必须是>=3的奇数，例如 3/5/7")
    kernel = np.ones((nms_ksize, nms_ksize), np.uint8)

    R_dilate = cv2.dilate(R_th, kernel)
    corners = (R_th == R_dilate) & (R_th > 0)

    ys, xs = np.where(corners)
    return R, corners, (xs, ys)


def draw_corners(bgr: np.ndarray, xs, ys, radius=3):
    out = bgr.copy()
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), int(radius), (0, 0, 255), 1)
    return out


#  4) 直方图均衡化 
def hist_equalize_gray(gray_u8: np.ndarray):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float32)
    cdf = np.cumsum(hist)

    cdf_nonzero = cdf[cdf > 0]
    cdf_min = float(cdf_nonzero.min()) if cdf_nonzero.size > 0 else 0.0
    cdf_max = float(cdf.max()) if cdf.size > 0 else 1.0

    lut = np.floor((cdf - cdf_min) / (cdf_max - cdf_min + 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    eq = lut[gray_u8]

    hist_eq = np.bincount(eq.ravel(), minlength=256).astype(np.float32)
    return eq, hist, hist_eq


#  可视化
def save_gray_image(title, img_float_or_u8, out_path, normalize=True):
    if img_float_or_u8.dtype != np.uint8 and normalize:
        vis = normalize_to_uint8(img_float_or_u8)
    else:
        vis = img_float_or_u8.astype(np.uint8) if img_float_or_u8.dtype != np.uint8 else img_float_or_u8

    plt.figure(figsize=(6, 6))
    plt.imshow(vis, cmap="gray")
    plt.title(title)
    plt.axis("off")
    save_fig(out_path)
    plt.close()


def save_hist_compare(img_before_u8, img_after_u8, hist_before, hist_after, out_path):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_before_u8, cmap="gray")
    plt.title("Before")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img_after_u8, cmap="gray")
    plt.title("After")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.plot(hist_before)
    plt.title("Histogram Before")
    plt.xlim([0, 255])

    plt.subplot(2, 2, 4)
    plt.plot(hist_after)
    plt.title("Histogram After")
    plt.xlim([0, 255])

    save_fig(out_path)
    plt.close()


def main():
    out_dir = "outputs"
    ensure_dir(out_dir)

    gray = read_gray_fixed()
    bgr = read_bgr_fixed()

    #  1) Sobel 
    gx, gy, mag, ang = sobel_gradients(gray)
    save_gray_image("Sobel |Gx|", np.abs(gx), os.path.join(out_dir, "sobel_gx.png"))
    save_gray_image("Sobel |Gy|", np.abs(gy), os.path.join(out_dir, "sobel_gy.png"))
    save_gray_image("Sobel Magnitude", mag, os.path.join(out_dir, "sobel_mag.png"))

    #  2) Canny 
    canny = canny_manual(gray, blur_ksize=5, blur_sigma=1.2, low_ratio=0.08, high_ratio=0.20)
    save_gray_image("Canny Step1: Gaussian Blur", canny["blurred"], os.path.join(out_dir, "canny_blur.png"))
    save_gray_image("Canny Step2: Gradient Magnitude", canny["mag"], os.path.join(out_dir, "canny_mag.png"))
    save_gray_image("Canny Step3: Non-Max Suppression", canny["nms"], os.path.join(out_dir, "canny_nms.png"), normalize=True)
    save_gray_image(
        f"Canny Result (Manual) low={canny['low']:.2f} high={canny['high']:.2f}",
        canny["edges"],
        os.path.join(out_dir, "canny_edges.png"),
        normalize=False
    )

    #  3) Harris 参数影响分析 
    win_sigmas = [1.0, 2.0, 3.0]
    nms_ksizes = [3, 5, 7]

    rows = []
    for sigma in win_sigmas:
        for nms_k in nms_ksizes:
            R, corners_mask, (xs, ys) = harris_manual(
                gray,
                k=0.04,
                win_sigma=sigma,
                thresh_ratio=0.01,
                nms_ksize=nms_k
            )

            save_gray_image(
                f"Harris Response R (sigma={sigma}, nms={nms_k})",
                R,
                os.path.join(out_dir, f"harris_R_sigma{sigma}_nms{nms_k}.png"),
                normalize=True
            )

            corners_vis = draw_corners(bgr, xs, ys, radius=3)
            plt.figure(figsize=(7, 7))
            plt.imshow(cv2.cvtColor(corners_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Harris Corners  sigma={sigma}, nms={nms_k}  count={len(xs)}")
            plt.axis("off")
            save_fig(os.path.join(out_dir, f"harris_corners_sigma{sigma}_nms{nms_k}.png"))
            plt.close()

            rows.append((sigma, nms_k, len(xs)))
            print(f"[Harris] sigma={sigma}, nms_ksize={nms_k}, corners={len(xs)}")

    csv_path = os.path.join(out_dir, "harris_param_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("win_sigma,nms_ksize,corners\n")
        for sigma, nms_k, cnt in rows:
            f.write(f"{sigma},{nms_k},{cnt}\n")
    print("[Saved]", csv_path)

    # 角点数量对比图
    plt.figure(figsize=(10, 5))
    plt.plot([r[2] for r in rows])
    plt.title("Harris Corner Count Sweep (order = sigma x nms_ksize)")
    plt.xlabel("Experiment Index")
    plt.ylabel("Corner Count")
    save_fig(os.path.join(out_dir, "harris_param_sweep_plot.png"))
    plt.close()

    #  4) 直方图均衡化 
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    eq, hist_before, hist_after = hist_equalize_gray(gray_u8)

    save_gray_image("Histogram Equalization - Before", gray_u8, os.path.join(out_dir, "hist_before.png"), normalize=False)
    save_gray_image("Histogram Equalization - After", eq, os.path.join(out_dir, "hist_after.png"), normalize=False)
    save_hist_compare(gray_u8, eq, hist_before, hist_after, os.path.join(out_dir, "hist_compare.png"))

    print("\nDone. All results saved to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()

