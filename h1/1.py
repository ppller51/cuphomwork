import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#  输出目录
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)

#读图（不用cv2）
def read_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def to_gray(img_rgb):
    img = img_rgb.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)

# 手写二维高斯核
def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0 or ksize <= 0:
        raise ValueError("ksize 必须是正奇数")
    if sigma <= 0:
        raise ValueError("sigma 必须 > 0")

    r = ksize // 2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    g = g / np.sum(g)
    return g.astype(np.float32)

# 手写 padding（至少两种）
def pad_image(img: np.ndarray, pad_h: int, pad_w: int, mode: str = "zero") -> np.ndarray:
    if pad_h < 0 or pad_w < 0:
        raise ValueError("pad must be non-negative")

    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError("img must be 2D or 3D")

    if C is None:
        out = np.zeros((H + 2*pad_h, W + 2*pad_w), dtype=img.dtype)
        out[pad_h:pad_h+H, pad_w:pad_w+W] = img
    else:
        out = np.zeros((H + 2*pad_h, W + 2*pad_w, C), dtype=img.dtype)
        out[pad_h:pad_h+H, pad_w:pad_w+W, :] = img

    if mode == "zero":
        return out

    if mode == "replicate":
        # 上下边
        out[:pad_h, pad_w:pad_w+W] = out[pad_h:pad_h+1, pad_w:pad_w+W]
        out[pad_h+H:, pad_w:pad_w+W] = out[pad_h+H-1:pad_h+H, pad_w:pad_w+W]
        # 左右边（含角）
        out[:, :pad_w] = out[:, pad_w:pad_w+1]
        out[:, pad_w+W:] = out[:, pad_w+W-1:pad_w+W]
        return out

    if mode == "reflect":
        # 反射（不重复边界点）：... c b | a b c | b a ...
        if pad_h > 0:
            out[:pad_h, pad_w:pad_w+W] = out[pad_h+1:pad_h+1+pad_h][::-1, pad_w:pad_w+W]
            out[pad_h+H:, pad_w:pad_w+W] = out[pad_h+H-1-pad_h:pad_h+H-1][::-1, pad_w:pad_w+W]
        if pad_w > 0:
            out[:, :pad_w] = out[:, pad_w+1:pad_w+1+pad_w][:, ::-1]
            out[:, pad_w+W:] = out[:, pad_w+W-1-pad_w:pad_w+W-1][:, ::-1]
        return out

    raise ValueError("mode must be 'zero', 'replicate', or 'reflect'")

# 手写卷积（不调用opencv滤波）
def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = "zero") -> np.ndarray:
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")
    kH, kW = kernel.shape
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError("kernel size must be odd")

    pad_h, pad_w = kH // 2, kW // 2
    kernel_flipped = kernel[::-1, ::-1].astype(np.float32)

    if img.ndim == 2:
        H, W = img.shape
        padded = pad_image(img, pad_h, pad_w, mode=padding).astype(np.float32)
        out = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kH, j:j+kW]
                out[i, j] = np.sum(patch * kernel_flipped)
        return out

    elif img.ndim == 3:
        H, W, C = img.shape
        padded = pad_image(img, pad_h, pad_w, mode=padding).astype(np.float32)
        out = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    patch = padded[i:i+kH, j:j+kW, c]
                    out[i, j, c] = np.sum(patch * kernel_flipped)
        return out

    else:
        raise ValueError("img must be 2D or 3D")

def clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

# 可视化（自动保存）
def show_kernels(kernels, titles, save_path=None, show=True, dpi=200):
    n = len(kernels)
    plt.figure(figsize=(4*n, 4))
    for i, (k, t) in enumerate(zip(kernels, titles), 1):
        plt.subplot(1, n, i)
        plt.imshow(k, cmap="viridis")
        plt.title(t)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()
    plt.close()

def show_images_grid(images, titles, cols=3, save_path=None, show=True, dpi=200):
    n = len(images)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for idx, (im, t) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, idx)
        if im.ndim == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(im)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()
    plt.close()


def main():
    img_path = r"F:\h1\test.png"
    out_dir = "outputs"
    ensure_dir(out_dir)

    img_rgb = read_image_rgb(img_path)
    img_gray = to_gray(img_rgb)

    #  三组参数：手写核 vs OpenCV 核对比
    combos = [(3, 0.8), (7, 1.5), (15, 3.0)]
    manual_kernels = [gaussian_kernel_2d(k, s) for k, s in combos]

    # OpenCV 生成对比核
    import cv2
    opencv_kernels = []
    diffs = []
    for (k, s), mk in zip(combos, manual_kernels):
        g1 = cv2.getGaussianKernel(k, s)  # (k,1)
        ok = (g1 @ g1.T).astype(np.float32)
        ok = ok / np.sum(ok)
        opencv_kernels.append(ok)
        diffs.append(np.max(np.abs(mk - ok)))

    show_kernels(
        manual_kernels,
        [f"Manual k={k}, σ={s}" for k, s in combos],
        save_path=os.path.join(out_dir, "kernels_manual.png"),
        show=True
    )

    show_kernels(
        opencv_kernels,
        [f"OpenCV k={k}, σ={s}" for k, s in combos],
        save_path=os.path.join(out_dir, "kernels_opencv.png"),
        show=True
    )

    for (k, s), d in zip(combos, diffs):
        print(f"[Kernel diff] k={k}, σ={s}, max|manual-opencv| = {d:.8f}")

    # 手写滤波：同一张图，不同核大小/σ 做高斯模糊对比
    blurred_imgs = []
    titles = []
    for (k, s) in combos:
        ker = gaussian_kernel_2d(k, s)
        out = convolve2d(img_gray, ker, padding="reflect")
        out_u8 = clip_uint8(out)
        blurred_imgs.append(out_u8)
        titles.append(f"Blur (manual) k={k}, σ={s}")

        Image.fromarray(out_u8).save(os.path.join(out_dir, f"blur_k{k}_sigma{s}.png"))

    show_images_grid(
        [clip_uint8(img_gray)] + blurred_imgs,
        ["Original Gray"] + titles,
        cols=2,
        save_path=os.path.join(out_dir, "blur_compare.png"),
        show=True
    )

    # padding 对比：至少两种 padding 的结果可视化
    k, s = 15, 3.0
    ker = gaussian_kernel_2d(k, s)

    pad_modes = ["zero", "replicate", "reflect"]  # 至少展示其中两种即可
    pad_results = []
    pad_titles = []
    for m in pad_modes:
        out = convolve2d(img_gray, ker, padding=m)
        out_u8 = clip_uint8(out)
        pad_results.append(out_u8)
        pad_titles.append(f"Padding={m}")

        Image.fromarray(out_u8).save(os.path.join(out_dir, f"padding_{m}_k{k}_sigma{s}.png"))

    show_images_grid(
        pad_results, pad_titles, cols=3,
        save_path=os.path.join(out_dir, "padding_blur_compare.png"),
        show=True
    )

    # padding 直接效果
    p = 30
    padded_demo = [pad_image(clip_uint8(img_gray), p, p, mode=m) for m in pad_modes]
    show_images_grid(
        padded_demo, [f"Padded view: {m}" for m in pad_modes], cols=3,
        save_path=os.path.join(out_dir, "padding_view_compare.png"),
        show=True
    )

    print(f"\nDone. 输出已保存到: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
