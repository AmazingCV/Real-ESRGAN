import argparse
import os
import glob
import time
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import lpips  # type: ignore
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def bgr2y(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.float32)
    img = img_bgr.astype(np.float32)
    # ITU-R BT.601 conversion
    y = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    return y


def compute_psnr(sr: np.ndarray, hr: np.ndarray, on_y: bool = False, crop_border: int = 0) -> float:
    if crop_border > 0:
        sr = sr[crop_border:-crop_border, crop_border:-crop_border, ...]
        hr = hr[crop_border:-crop_border, crop_border:-crop_border, ...]
    if on_y:
        sr = bgr2y(sr)
        hr = bgr2y(hr)
    else:
        # to float32
        sr = sr.astype(np.float32)
        hr = hr.astype(np.float32)
    diff = sr.astype(np.float32) - hr.astype(np.float32)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    PIX_MAX = 255.0
    return 10.0 * math.log10((PIX_MAX * PIX_MAX) / mse)


def compute_ssim(sr: np.ndarray, hr: np.ndarray, on_y: bool = False, crop_border: int = 0) -> Optional[float]:
    if not _HAS_SKIMAGE:
        return None
    if crop_border > 0:
        sr = sr[crop_border:-crop_border, crop_border:-crop_border, ...]
        hr = hr[crop_border:-crop_border, crop_border:-crop_border, ...]
    if on_y:
        sr = bgr2y(sr).astype(np.uint8)
        hr = bgr2y(hr).astype(np.uint8)
        return float(structural_similarity(sr, hr, data_range=255))
    else:
        # multichannel SSIM on BGR
        return float(structural_similarity(sr, hr, channel_axis=-1, data_range=255))


def compute_lpips(sr: np.ndarray, hr: np.ndarray, lpips_model, device: torch.device) -> Optional[float]:
    if not _HAS_LPIPS:
        return None
    # Convert BGR [0,255] to RGB [-1,1]
    sr_rgb = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hr_rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sr_t = torch.from_numpy(sr_rgb).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    hr_t = torch.from_numpy(hr_rgb).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    sr_t = sr_t.to(device)
    hr_t = hr_t.to(device)
    with torch.no_grad():
        d = lpips_model(sr_t, hr_t)
    return float(d.item())


def map_hr_path(lr_path: str, hr_dir: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(lr_path))[0]
    candidates = []
    for e in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']:
        p = os.path.join(hr_dir, base + e)
        if os.path.isfile(p):
            candidates.append(p)
    if candidates:
        # Prefer png/jpg
        for pref in ['.png', '.jpg', '.jpeg']:
            for p in candidates:
                if p.lower().endswith(pref):
                    return p
        return candidates[0]
    return None


def forward_only(upsampler: RealESRGANer, img_bgr: np.ndarray, outscale: Optional[float] = None,
                 alpha_upsampler: str = 'realesrgan') -> Tuple[np.ndarray, float]:
    """Run forward pass timing-only (exclude model init and disk I/O).
    Returns (output_bgr_uint8, forward_time_seconds).
    """
    h_input, w_input = img_bgr.shape[0:2]
    img = img_bgr.astype(np.float32)
    max_range = 65535 if np.max(img) > 256 else 255

    # detect mode and convert to RGB if needed
    if img.ndim == 2:
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if alpha_upsampler == 'realesrgan':
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / float(max_range)

    # pre-process
    upsampler.pre_process(img)

    # forward timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_t = time.perf_counter()
    if upsampler.tile_size > 0:
        upsampler.tile_process()
    else:
        upsampler.process()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_t

    # post-process
    output_img = upsampler.post_process()
    output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    # alpha channel handling
    if img_mode == 'L':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    elif img_mode == 'RGBA':
        if alpha_upsampler == 'realesrgan':
            upsampler.pre_process(alpha / float(max_range))
            if upsampler.tile_size > 0:
                upsampler.tile_process()
            else:
                upsampler.process()
            output_alpha = upsampler.post_process()
            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        else:
            h, w = alpha.shape[0:2]
            output_alpha = cv2.resize(alpha, (w * upsampler.scale, h * upsampler.scale), interpolation=cv2.INTER_LINEAR)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    if max_range == 65535:
        output = (output_img * 65535.0).round().astype(np.uint16)
    else:
        output = (output_img * 255.0).round().astype(np.uint8)

    if outscale is not None and outscale != float(upsampler.scale):
        output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_LANCZOS4)

    return output, elapsed


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN x2 批量推理与指标评估')
    parser.add_argument('--model_path', type=str, required=False,
                        default='/mnt/alg/home/tanggz/Real-ESRGAN-master/experiments/finetune_MyDataset_x2plus/models/net_g_280000.pth',
                        help='生成器权重路径（建议使用 net_g_*.pth）')
    parser.add_argument('--lr_dir', type=str, required=False,
                        default='/mnt/alg/home/tanggz/Real-ESRGAN-master/inputs/test_200/LR',
                        help='LR 图像目录')
    parser.add_argument('--hr_dir', type=str, required=False,
                        default='/mnt/alg/home/tanggz/Real-ESRGAN-master/inputs/test_200/HR',
                        help='HR 图像目录（用于指标评估）')
    parser.add_argument('--out_dir', type=str, required=False,
                        default='/mnt/alg/home/tanggz/Real-ESRGAN-master/results/Real-ESRGAN_200张',
                        help='输出目录')
    parser.add_argument('--time_txt', type=str, required=False,
                        default='/mnt/alg/home/tanggz/Real-ESRGAN-master/results/Real-ESRGAN_200张/推理时间.txt',
                        help='推理时间与汇总结果输出文件')
    parser.add_argument('--tile', type=int, default=0)
    parser.add_argument('--tile_pad', type=int, default=10)
    parser.add_argument('--pre_pad', type=int, default=0)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--eval_on_y', action='store_true', help='在Y通道计算PSNR/SSIM')
    parser.add_argument('--crop_border', type=int, default=0, help='在计算指标前裁边像素')

    args = parser.parse_args()

    # 构建模型（RealESRGAN_x2plus 架构）
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    try:
        upsampler = RealESRGANer(
            scale=2,
            model_path=args.model_path,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            gpu_id=args.gpu_id
        )
    except Exception as e:
        raise RuntimeError(
            f'加载模型失败: {e}. 请确认提供的是生成器权重 (例如 net_g_*.pth), 而非判别器权重 (net_d_*.pth)')

    device = upsampler.device

    # LPIPS 模型
    lpips_model = None
    if _HAS_LPIPS:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()

    # 路径检查
    if not os.path.isdir(args.lr_dir):
        raise FileNotFoundError(f'LR 目录不存在: {args.lr_dir}')
    if not os.path.isdir(args.hr_dir):
        print(f'警告: HR 目录不存在，跳过指标评估 -> {args.hr_dir}')
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.time_txt))

    lr_paths = list_images(args.lr_dir)
    if len(lr_paths) == 0:
        raise RuntimeError(f'在 {args.lr_dir} 未找到图像')

    per_image_records = []  # (name, psnr, ssim, lpips, time_s)

    for idx, lr_path in enumerate(lr_paths):
        name = os.path.splitext(os.path.basename(lr_path))[0]
        print(f'[{idx+1}/{len(lr_paths)}] 推理: {name}')
        img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'跳过（无法读取）: {lr_path}')
            continue

        with torch.no_grad():
            sr, t_forward = forward_only(upsampler, img, outscale=2.0)

        save_ext = '.png' if (sr.ndim == 3 and sr.shape[2] == 4) else '.jpg'
        save_path = os.path.join(args.out_dir, f'{name}_x2{save_ext}')
        cv2.imwrite(save_path, sr)

        # 指标
        psnr_v, ssim_v, lpips_v = None, None, None
        hr_path = map_hr_path(lr_path, args.hr_dir) if os.path.isdir(args.hr_dir) else None
        if hr_path and os.path.isfile(hr_path):
            hr = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
            if hr is not None:
                if hr.shape[:2] != sr.shape[:2]:
                    print(f'警告: HR 与 SR 尺寸不匹配，调整 SR 到 HR 尺寸再评估: {name}')
                    sr_eval = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
                else:
                    sr_eval = sr
                psnr_v = compute_psnr(sr_eval, hr, on_y=args.eval_on_y, crop_border=args.crop_border)
                ssim_v = compute_ssim(sr_eval, hr, on_y=args.eval_on_y, crop_border=args.crop_border)
                if _HAS_LPIPS and lpips_model is not None:
                    lpips_v = compute_lpips(sr_eval, hr, lpips_model, device)
            else:
                print(f'警告: 无法读取 HR: {hr_path}')
        else:
            if hr_path is None:
                print(f'未找到对应 HR（同名文件）: {name}.* 于 {args.hr_dir}')

        per_image_records.append((name, psnr_v, ssim_v, lpips_v, t_forward))

    # 汇总
    times = [t for (_, _, _, _, t) in per_image_records]
    avg_time = float(np.mean(times)) if times else float('nan')
    fps = (1.0 / avg_time) if (avg_time and not math.isnan(avg_time) and avg_time > 0) else float('nan')

    # 写结果
    with open(args.time_txt, 'w', encoding='utf-8') as f:
        f.write('文件名, PSNR(dB), SSIM, LPIPS, 推理时间(ms)\n')
        for (name, p, s, l, t) in per_image_records:
            ms = t * 1000.0
            p_str = f'{p:.4f}' if isinstance(p, float) else '-'
            s_str = f'{s:.6f}' if isinstance(s, float) else '-'
            l_str = f'{l:.6f}' if isinstance(l, float) else '-'
            f.write(f'{name}, {p_str}, {s_str}, {l_str}, {ms:.3f}\n')
        f.write('\n')
        f.write(f'平均推理时间: {avg_time * 1000.0:.3f} ms\n')
        f.write(f'平均帧率: {fps:.3f} FPS\n')

    print('完成。')


if __name__ == '__main__':
    main()
