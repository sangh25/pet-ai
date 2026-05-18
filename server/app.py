# -*- coding: utf-8 -*-
"""
Pet-AI Dog Emotion App - Public Server Version
- C:\\pet-ai\\server 또는 공용 서버 전용
- 빠른모드만 사용
- YOLO는 yolo26n.pt 고정
- baseline / HuFEP 선택
- 이미지 / 동영상 / 사진 찍기 / 웹캠 실시간 지원

모델 파일 위치:
  C:\\pet-ai\\server\\models\\baseline_best_val_f1.pt
  C:\\pet-ai\\server\\models\\hufep_best_val_f1.pt
  C:\\pet-ai\\server\\models\\yolo26n.pt

선택 모델:
  baseline_best_val_f1.pt
  hufep_best_val_f1.pt
  yolo26n.pt
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO

# =========================================================
# 0. 기본 설정
# =========================================================
torch.set_grad_enabled(False)

BASE_DIR = Path(os.environ.get("PET_AI_BASE", Path(__file__).resolve().parent)).resolve()
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE_ENV = os.environ.get("PET_AI_DEVICE", "cpu").lower().strip()
if DEVICE_ENV in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
try:
    torch.set_num_threads(max(1, int(os.environ.get("PET_AI_THREADS", "4"))))
except Exception:
    pass

CLASS_NAMES = ["angry", "happy", "relaxed", "sad"]
CLASS_KO = {"angry": "분노", "happy": "행복", "relaxed": "편안", "sad": "슬픔"}
COCO_DOG_CLASS_ID = 16
DEBUG = os.environ.get("PET_AI_DEBUG", "1").lower() not in {"0", "false", "no", "off"}

FAST_YOLO_PATH = MODEL_DIR / "yolo26n.pt"
ANALYSIS_YOLO_PATH = FAST_YOLO_PATH

CHECKPOINTS: Dict[str, Dict[str, List[Path]]] = {
    "fast": {
        "baseline": [MODEL_DIR / "baseline_best_val_f1.pt"],
        "hufep": [MODEL_DIR / "hufep_best_val_f1.pt"],
    },
}

MODE_CONFIGS = {
    "fast": {
        "title": "빠른모드",
        "yolo_name": "yolo26n.pt",
        "yolo_path": FAST_YOLO_PATH,
        "imgsz": int(os.environ.get("PET_AI_FAST_YOLO_IMGSZ", "416")),
        "confs": [0.10, 0.08, 0.05, 0.03],
        "iou": float(os.environ.get("PET_AI_FAST_YOLO_IOU", "0.55")),
        # 공용 서버 고정: 최대 3마리까지만 분석합니다.
        "max_dogs": min(3, max(1, int(os.environ.get("PET_AI_FAST_MAX_DOGS", "3")))),
        "emotion_profile": "weak",
        "video_stride": int(os.environ.get("PET_AI_FAST_VIDEO_STRIDE", "5")),
    },
}

ENGINE_CACHE: Dict[str, Dict[str, Any]] = {}
DETECTOR_CACHE: Dict[str, YOLO] = {}


def log(*args):
    if DEBUG:
        print("[PetAI]", *args, flush=True)


def normalize_mode(mode: str) -> str:
    # 공용 서버 버전: 모든 요청을 빠른 설정으로 처리합니다.
    return "fast"


def normalize_model(model: str) -> str:
    v = str(model or "hufep").lower().strip()
    if v in {"base", "baseline", "vit", "1"}:
        return "baseline"
    # 공용 서버 lite 버전은 baseline / HuFEP 두 모델만 선택합니다.
    return "hufep"


# app-4 UI compatibility names
MODEL_DISPLAY_NAMES = {
    "baseline": "baseline",
    "hufep": "HuFEP",
}

def normalize_model_choice(model_choice: str = "hufep") -> str:
    return normalize_model(model_choice)


def safe_torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def get_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(torch.is_tensor(v) for v in ckpt.values() if v is not None):
            return ckpt
    return ckpt if isinstance(ckpt, dict) else {}


def clean_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = str(k)
        for prefix in ["module.", "model.", "backbone."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # crop-fusion 학습 파일이어도 ViT backbone 부분만 최대한 로드
        if nk.startswith("vit."):
            nk = nk[4:]
        if nk.startswith("encoder."):
            nk = nk[8:]
        out[nk] = v
    return out


def guess_img_size(state: Dict[str, Any], default: int = 336) -> int:
    pe = state.get("pos_embed")
    if pe is not None and hasattr(pe, "shape"):
        try:
            n = int(pe.shape[1]) - 1
            side = int(round(math.sqrt(n)))
            if side * side == n:
                # ViT 위치 임베딩 기준. patch16 모델이면 config img_size가 우선됨.
                return int(side * 14)
        except Exception:
            pass
    return int(default)


def checkpoint_classes(ckpt: Any) -> List[str]:
    if isinstance(ckpt, dict) and "idx_to_class" in ckpt:
        try:
            m = {int(k): str(v) for k, v in ckpt["idx_to_class"].items()}
            return [m[i] for i in range(len(m))]
        except Exception:
            pass
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        return [str(x) for x in ckpt["classes"]]
    return list(CLASS_NAMES)


def candidate_model_names(model_name: str) -> List[str]:
    out = []
    if model_name:
        out.append(model_name)
        if model_name.endswith(".openai"):
            out.append(model_name[:-7])
    out += [
        "vit_huge_patch14_clip_224",
        "vit_huge_patch14_224",
        "vit_base_patch16_384.augreg_in1k",
        "vit_base_patch16_224.augreg_in21k",
        "vit_base_patch16_224",
    ]
    ret = []
    for x in out:
        if x and x not in ret:
            ret.append(x)
    return ret


def create_timm_model(model_name: str, num_classes: int, img_size_val: int):
    last = None
    for cand in candidate_model_names(model_name):
        try:
            try:
                model = timm.create_model(cand, pretrained=False, num_classes=num_classes, img_size=img_size_val)
            except Exception:
                model = timm.create_model(cand, pretrained=False, num_classes=num_classes)
            return model, cand
        except Exception as e:
            last = e
    raise RuntimeError(f"timm 모델 생성 실패: {model_name} / {last}")


def load_engine(path: Path) -> Dict[str, Any]:
    cache_key = str(path.resolve())
    if cache_key in ENGINE_CACHE:
        return ENGINE_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")

    ckpt = safe_torch_load(path)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state = clean_state_dict(get_state_dict(ckpt))
    classes = checkpoint_classes(ckpt)
    model_name = str(cfg.get("model_name") or os.environ.get("PET_AI_TIMM_MODEL") or "vit_huge_patch14_clip_224")
    img_size_val = int(cfg.get("img_size") or os.environ.get("PET_AI_IMG_SIZE") or guess_img_size(state, 224))

    model, actual_name = create_timm_model(model_name, len(classes), img_size_val)
    try:
        model.load_state_dict(state, strict=True)
        strict_used = True
    except Exception as e:
        log("strict=True 실패 -> strict=False 로드:", path.name, repr(e)[:500])
        missing, unexpected = model.load_state_dict(state, strict=False)
        strict_used = False
        if len(missing) > 50:
            log("missing keys many:", len(missing))
        if len(unexpected) > 50:
            log("unexpected keys many:", len(unexpected))

    model = model.to(DEVICE).eval()
    tfm = T.Compose([
        T.Resize((img_size_val, img_size_val), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    engine = {
        "path": str(path),
        "name": path.name,
        "model": model,
        "classes": classes,
        "img_size": img_size_val,
        "transform": tfm,
        "timm_name": actual_name,
        "strict": strict_used,
    }
    ENGINE_CACHE[cache_key] = engine
    log(f"Loaded model {path.name} | timm={actual_name} | img={img_size_val} | classes={classes} | strict={strict_used}")
    return engine


def select_model_paths(mode: str, model_choice: str) -> List[Path]:
    mode = normalize_mode(mode)
    model_choice = normalize_model(model_choice)
    paths: List[Path] = []
    if model_choice == "baseline":
        paths += CHECKPOINTS[mode]["baseline"]
    elif model_choice == "hufep":
        paths += CHECKPOINTS[mode]["hufep"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("모델 파일이 없습니다: " + ", ".join(missing) + f" / 위치: {MODEL_DIR}")
    return paths


def get_detector(mode: str) -> YOLO:
    mode = normalize_mode(mode)
    cfg = MODE_CONFIGS[mode]
    path = cfg["yolo_path"]
    if not path.exists():
        raise FileNotFoundError(f"YOLO 파일이 없습니다: {path}")
    key = str(path.resolve())
    if key not in DETECTOR_CACHE:
        log("Loading YOLO:", path)
        DETECTOR_CACHE[key] = YOLO(str(path))
    return DETECTOR_CACHE[key]


def is_dog_class(detector_model: YOLO, cls_id: int) -> bool:
    cls_id = int(cls_id)
    if cls_id == COCO_DOG_CLASS_ID:
        return True
    try:
        names = detector_model.names
        if isinstance(names, dict):
            return str(names.get(cls_id, "")).lower() == "dog"
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id]).lower() == "dog"
    except Exception:
        pass
    return False


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[int]:
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return [x1, y1, x2, y2]


def pad_box(box: List[float], w: int, h: int, pad: float = 0.10, square: bool = False) -> List[int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    if square:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(bw, bh) * (1 + 2 * pad)
        return clip_box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, w, h)
    return clip_box(x1 - bw * pad, y1 - bh * pad, x2 + bw * pad, y2 + bh * pad, w, h)


def upper_head_box(box: List[float], w: int, h: int, pad: float = 0.10) -> List[int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    nx1, nx2 = x1 + bw * 0.06, x2 - bw * 0.06
    ny1, ny2 = y1, y1 + bh * 0.70
    return pad_box([nx1, ny1, nx2, ny2], w, h, pad=pad, square=True)


def face_safe_box(box: List[float], w: int, h: int, pad: float = 0.14) -> List[int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    nx1, nx2 = x1 + bw * 0.12, x2 - bw * 0.12
    ny1, ny2 = y1 + bh * 0.02, y1 + bh * 0.62
    return pad_box([nx1, ny1, nx2, ny2], w, h, pad=pad, square=True)


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (box_area(a) + box_area(b) - inter + 1e-9)


def small_overlap(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (min(box_area(a), box_area(b)) + 1e-9)


def box_intersection(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_distance_ratio(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    # 같은 강아지에서 나온 중복 박스는 중심이 가깝고, 서로 다른 강아지는 중심이 더 멀다.
    scale = max(1.0, min(max(aw, ah), max(bw, bh)))
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale)


def is_duplicate_dog_box(a: List[float], b: List[float]) -> bool:
    """붙어 있는 여러 강아지는 살리고, 같은 강아지의 얼굴/몸 일부 중복 박스만 제거."""
    iou = box_iou(a, b)
    contain = small_overlap(a, b)
    cdist = center_distance_ratio(a, b)
    area_a, area_b = box_area(a), box_area(b)
    area_ratio = min(area_a, area_b) / max(area_a, area_b, 1e-9)

    # 거의 같은 박스
    if iou >= 0.62:
        return True
    # 작은 박스가 큰 박스 안에 많이 들어간 경우: 얼굴/몸 일부 중복일 가능성이 큼
    if contain >= 0.74 and cdist <= 0.72:
        return True
    # 크기가 많이 다른데 꽤 겹치면 작은 부분 박스 제거
    if contain >= 0.58 and area_ratio <= 0.55 and cdist <= 0.95:
        return True
    return False


def dedupe_dets(dets: List[Dict[str, Any]], max_dogs: int) -> List[Dict[str, Any]]:
    # 신뢰도만 우선하면 얼굴 일부 박스가 먼저 남을 수 있어서, 신뢰도+면적을 같이 봅니다.
    dets = sorted(dets, key=lambda d: (float(d.get("det_conf", 0)), float(d.get("area_ratio", 0)) * 0.25), reverse=True)
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            if is_duplicate_dog_box(d["raw_box"], k["raw_box"]):
                dup = True
                break
        if not dup:
            kept.append(d)
        if len(kept) >= max_dogs:
            break
    kept.sort(key=lambda d: (d["box"][0], d["box"][1]))
    return kept


def detect_dogs(img: Image.Image, mode: str) -> List[Dict[str, Any]]:
    mode = normalize_mode(mode)
    cfg = MODE_CONFIGS[mode]
    detector = get_detector(mode)
    w, h = img.size
    all_dets: List[Dict[str, Any]] = []
    attempts = []
    for c in cfg["confs"]:
        attempts.append((float(c), float(cfg["iou"]), int(cfg["imgsz"])))
    # 작거나 구석 강아지 보완. 단, 해당 모드 YOLO만 사용한다.
    if mode == "fast":
        attempts += [(0.05, 0.45, 960)]
    else:
        attempts += [(0.03, 0.55, 960), (0.03, 0.65, 768), (0.01, 0.55, 1280)]

    seen = set()
    unique_attempts = []
    for c, iou, imgsz in attempts:
        key = (round(c, 4), round(iou, 4), int(imgsz))
        if key not in seen:
            seen.add(key)
            unique_attempts.append((c, iou, imgsz))

    log("DETECT", mode, cfg["yolo_name"], "attempts", unique_attempts)
    for conf, iou, imgsz in unique_attempts:
        result = detector.predict(
            source=np.array(img.convert("RGB")),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            agnostic_nms=False,
            max_det=max(80, int(cfg["max_dogs"]) * 16),
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        clss = result.boxes.cls.detach().cpu().numpy().astype(int)
        raw_dog = 0
        kept_this = 0
        for box, det_conf, cls_id in zip(boxes, confs, clss):
            if not is_dog_class(detector, int(cls_id)):
                continue
            raw_dog += 1
            raw = [float(x) for x in box]
            area_ratio = box_area(raw) / max(1.0, float(w * h))
            if area_ratio < 0.00035 or area_ratio > 0.990:
                continue
            aspect = max((raw[2] - raw[0]) / max(1, raw[3] - raw[1]), (raw[3] - raw[1]) / max(1, raw[2] - raw[0]))
            if aspect > 4.0:
                continue
            all_dets.append({
                "raw_box": raw,
                "box": pad_box(raw, w, h, pad=0.08, square=False),
                "det_conf": float(det_conf),
                "area_ratio": float(area_ratio),
                "class_id": int(cls_id),
                "imgsz": int(imgsz),
                "yolo_conf_used": float(conf),
                "yolo": cfg["yolo_name"],
            })
            kept_this += 1
        log(f"  {cfg['yolo_name']} imgsz={imgsz} conf={conf} iou={iou} raw_dog={raw_dog} kept={kept_this}")
        if len(dedupe_dets(all_dets, int(cfg["max_dogs"]))) >= int(cfg["max_dogs"]):
            break

    final = dedupe_dets(all_dets, int(cfg["max_dogs"]))

    # =========================================================
    # FAST mode YOLO last retry
    # - yolo26n 빠른모드에서 작은/구석 강아지를 놓치는 경우 보완
    # - 감정 분석/CropFusion은 건드리지 않음
    # - 기존 YOLO 결과가 0마리일 때만 마지막 1회 재시도
    # =========================================================
    if len(final) == 0 and mode == "fast":
        try:
            log("FAST YOLO last retry: yolo26n conf=0.01 iou=0.45 imgsz=1280")
            result = detector.predict(
                source=np.array(img.convert("RGB")),
                conf=0.01,
                iou=0.45,
                imgsz=1280,
                agnostic_nms=False,
                max_det=max(120, int(cfg["max_dogs"]) * 24),
                verbose=False,
            )[0]
            retry_dets: List[Dict[str, Any]] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                clss = result.boxes.cls.detach().cpu().numpy().astype(int)
                raw_dog = 0
                kept_this = 0
                for box, det_conf, cls_id in zip(boxes, confs, clss):
                    if not is_dog_class(detector, int(cls_id)):
                        continue
                    raw_dog += 1
                    raw = [float(x) for x in box]
                    area_ratio = box_area(raw) / max(1.0, float(w * h))
                    # 마지막 재시도만 조금 더 약하게: 작은 이미지/구석 강아지 허용
                    if area_ratio < 0.00015 or area_ratio > 0.990:
                        continue
                    aspect = max(
                        (raw[2] - raw[0]) / max(1, raw[3] - raw[1]),
                        (raw[3] - raw[1]) / max(1, raw[2] - raw[0]),
                    )
                    if aspect > 4.5:
                        continue
                    retry_dets.append({
                        "raw_box": raw,
                        "box": pad_box(raw, w, h, pad=0.08, square=False),
                        "det_conf": float(det_conf),
                        "area_ratio": float(area_ratio),
                        "class_id": int(cls_id),
                        "imgsz": 1280,
                        "yolo_conf_used": 0.01,
                        "yolo": cfg["yolo_name"] + " last-retry",
                    })
                    kept_this += 1
                log(f"  yolo26n last-retry imgsz=1280 conf=0.01 iou=0.45 raw_dog={raw_dog} kept={kept_this}")
            final = dedupe_dets(retry_dets, int(cfg["max_dogs"]))
        except Exception as e:
            log("FAST YOLO last retry failed:", repr(e))

    # =========================================================
    # FAST mode yolo26n-only UPSCALE retry
    # - 빠른모드는 끝까지 yolo26n.pt만 사용합니다.
    # - yolo26n emergency 재시도는 사용하지 않습니다.
    # - 작은/구석/배경 섞인 강아지를 보완하기 위해, 0마리일 때만
    #   이미지를 2배 확대해서 yolo26n으로 한 번 더 감지합니다.
    # - 단, conf 0.005처럼 너무 낮은 값은 신발/바닥 오탐이 생겨 사용하지 않습니다.
    # - 확대 이미지에서 나온 box는 원본 좌표로 다시 환산합니다.
    # =========================================================
    if len(final) == 0 and mode == "fast" and os.environ.get("PET_AI_FAST_UPSCALE_RETRY", "1").lower() not in {"0", "false", "no", "off"}:
        try:
            scale = float(os.environ.get("PET_AI_FAST_UPSCALE_FACTOR", "2.0"))
            if scale < 1.1:
                scale = 2.0
            up_w = max(32, int(round(w * scale)))
            up_h = max(32, int(round(h * scale)))
            up_img = img.convert("RGB").resize((up_w, up_h), Image.Resampling.BICUBIC)
            upscale_attempts = [(0.02, 0.45, 1280), (0.015, 0.50, 1280), (0.015, 0.55, 1536)]
            upscale_dets: List[Dict[str, Any]] = []
            log(f"FAST yolo26n upscale retry: scale={scale} size={up_w}x{up_h}")
            for uconf, uiou, uimgsz in upscale_attempts:
                result = detector.predict(
                    source=np.array(up_img),
                    conf=float(uconf),
                    iou=float(uiou),
                    imgsz=int(uimgsz),
                    agnostic_nms=False,
                    max_det=max(160, int(cfg["max_dogs"]) * 32),
                    verbose=False,
                )[0]
                raw_dog = 0
                kept_this = 0
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
                    for box, det_conf, cls_id in zip(boxes, confs, clss):
                        if not is_dog_class(detector, int(cls_id)):
                            continue
                        raw_dog += 1
                        # upscaled 좌표 -> 원본 좌표
                        raw = [float(box[0]) / scale, float(box[1]) / scale, float(box[2]) / scale, float(box[3]) / scale]
                        raw = [float(x) for x in clip_box(raw[0], raw[1], raw[2], raw[3], w, h)]
                        area_ratio = box_area(raw) / max(1.0, float(w * h))
                        if area_ratio < 0.00018 or area_ratio > 0.995:
                            continue
                        aspect = max(
                            (raw[2] - raw[0]) / max(1, raw[3] - raw[1]),
                            (raw[3] - raw[1]) / max(1, raw[2] - raw[0]),
                        )
                        if aspect > 4.8:
                            continue

                        # upscale-retry는 오탐 방지를 위해 너무 낮은 confidence를 금지합니다.
                        # 예: 신발/바닥 물체를 dog로 착각하는 케이스 차단.
                        if float(det_conf) < 0.015:
                            continue
                        bx1, by1, bx2, by2 = raw
                        box_cy = (by1 + by2) / 2.0
                        box_h = max(1.0, by2 - by1)
                        # 작은 박스가 이미지 하단에 몰려 있으면 신발/바닥 오탐 가능성이 높아서 제거.
                        if box_cy > h * 0.62 and box_h < h * 0.42 and area_ratio < 0.22:
                            continue

                        upscale_dets.append({
                            "raw_box": raw,
                            "box": pad_box(raw, w, h, pad=0.08, square=False),
                            "det_conf": float(det_conf),
                            "area_ratio": float(area_ratio),
                            "class_id": int(cls_id),
                            "imgsz": int(uimgsz),
                            "yolo_conf_used": float(uconf),
                            "yolo": cfg["yolo_name"] + " upscale-retry",
                        })
                        kept_this += 1
                log(f"  yolo26n upscale-retry imgsz={uimgsz} conf={uconf} iou={uiou} raw_dog={raw_dog} kept={kept_this}")
                final = dedupe_dets(upscale_dets, int(cfg["max_dogs"]))
                if len(final) > 0:
                    break
        except Exception as e:
            log("FAST yolo26n upscale retry failed:", repr(e))

    # =========================================================
    # FAST mode rule
    # - 빠른모드는 무조건 yolo26n.pt만 사용합니다.
    # - yolo26n 기본/고해상도/확대 재시도까지 실패해도 yolo26n는 쓰지 않습니다.
    # - 그래도 감정 분석은 아래 fallback에서 전체 이미지 1마리 기준으로 가능합니다.
    # =========================================================
    log("DETECT FINAL", mode, "dogs=", len(final))
    for i, d in enumerate(final, 1):
        log(" dog", i, d["yolo"], "conf", round(d["det_conf"], 4), "box", d["box"])
    return final


def fixed_probs(raw_probs: np.ndarray, engine_classes: List[str]) -> np.ndarray:
    by_name = {str(engine_classes[i]): float(raw_probs[i]) for i in range(len(engine_classes))}
    arr = np.array([by_name.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)
    if float(arr.sum()) > 0:
        arr = arr / float(arr.sum())
    return arr


@torch.inference_mode()
def predict_one_crop(img: Image.Image, engine: Dict[str, Any]) -> np.ndarray:
    x = engine["transform"](img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        logits = engine["model"](x)
    raw = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
    return fixed_probs(raw, engine["classes"])


def relaxed_open_mouth_visual_suspect(img: Optional[Image.Image], raw_box: Optional[List[float]]) -> Dict[str, Any]:
    """relaxed 과확신인데 실제로 입벌림/이빨/불편 표정일 수 있는 근접 얼굴 패턴 감지.

    - 감정 확률은 절대 수정하지 않습니다.
    - 화면 표시용 guard에만 사용합니다.
    - 간단한 이미지 통계만 사용해서 별도 모델은 필요 없습니다.
    """
    info = {
        "suspect": False,
        "dark_ratio": 0.0,
        "very_dark_ratio": 0.0,
        "box_area_ratio": 0.0,
        "reason": "",
    }
    if img is None:
        return info
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if raw_box is None:
            # fallback/전체 이미지일 때는 중앙 얼굴 영역만 봅니다.
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = [int(v) for v in pad_box(raw_box, w, h, pad=0.04, square=False)]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        info["box_area_ratio"] = float((bw * bh) / max(1, w * h))

        crop = rgb.crop((x1, y1, x2, y2)).resize((224, 224))
        arr = np.asarray(crop).astype(np.float32)
        # luminance. 입/코/입안/이빨 주변의 어두운 영역을 보기 위한 단순 통계.
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # 강아지 입은 보통 얼굴 하단 중앙에 있으므로 그 구역만 확인.
        roi = lum[int(224 * 0.42):int(224 * 0.88), int(224 * 0.20):int(224 * 0.82)]
        dark_ratio = float((roi < 55).mean())
        very_dark_ratio = float((roi < 35).mean())
        info["dark_ratio"] = dark_ratio
        info["very_dark_ratio"] = very_dark_ratio

        # 너무 작은 박스에서는 오탐이 커서 적용하지 않음.
        close_or_large = info["box_area_ratio"] >= 0.22 or raw_box is None
        suspect = close_or_large and (dark_ratio >= 0.075 or very_dark_ratio >= 0.035)
        if suspect:
            info["suspect"] = True
            info["reason"] = f"large/close face with dark lower-mouth region: dark={dark_ratio:.3f}, very_dark={very_dark_ratio:.3f}"
        return info
    except Exception as e:
        info["reason"] = f"visual guard failed: {repr(e)[:120]}"
        return info


def fair_identity_emotion_meta(probs: np.ndarray, img: Optional[Image.Image] = None, raw_box: Optional[List[float]] = None) -> Dict[str, Any]:
    """4가지 감정 확률은 그대로 두고, 화면 표시만 안전하게 보정합니다.

    중요:
    - angry/happy/relaxed/sad 확률값은 절대 바꾸지 않습니다.
    - 최종 raw label도 그대로 둡니다.
    - 입 다문 찡그림형 화남이 happy로 과확신되는 케이스만
      "행복? / 분노 의심"으로 표시합니다.
    - 확률 차이가 작은 경우에는 top-2 애매함으로 표시합니다.
    """
    arr = np.asarray(probs, dtype=np.float32).copy()
    arr = arr / max(1e-9, float(arr.sum()))
    order = list(np.argsort(-arr))
    top1, top2 = int(order[0]), int(order[1])
    top1_name = CLASS_NAMES[top1]
    top2_name = CLASS_NAMES[top2]
    top1_p = float(arr[top1])
    top2_p = float(arr[top2])
    margin = top1_p - top2_p

    angry = float(arr[CLASS_NAMES.index("angry")])
    happy = float(arr[CLASS_NAMES.index("happy")])
    relaxed = float(arr[CLASS_NAMES.index("relaxed")])
    sad = float(arr[CLASS_NAMES.index("sad")])

    top2_items = [
        {
            "label": CLASS_NAMES[int(i)],
            "label_ko": CLASS_KO.get(CLASS_NAMES[int(i)], CLASS_NAMES[int(i)]),
            "confidence": float(arr[int(i)]),
        }
        for i in order[:2]
    ]

    display_label = top1_name
    display_label_ko = CLASS_KO.get(top1_name, top1_name)
    display_confidence = top1_p
    ambiguous = False
    emotion_note = ""
    guard_type = "none"

    # 1) 확신이 낮거나 1등/2등 차이가 작으면 단정하지 않음.
    if top1_p < 0.45 or margin < 0.08:
        ambiguous = True
        display_label = f"{top1_name}/{top2_name}"
        display_label_ko = f"{CLASS_KO.get(top1_name, top1_name)} / {CLASS_KO.get(top2_name, top2_name)} 애매함"
        emotion_note = "확률 차이가 작아서 한 감정으로 단정하지 않습니다."
        guard_type = "low_confidence_or_small_margin"

    # 2) 입 다문 찡그림형 화남이 happy로 과확신되는 케이스.
    #    실제 확률은 그대로 두고 표시만 "행복? / 분노 의심"으로 바꿉니다.
    closed_mouth_angry_like = (
        top1_name == "happy"
        and happy >= 0.60
        and sad >= 0.12
        and relaxed <= 0.18
        and angry <= 0.15
    )
    if closed_mouth_angry_like:
        ambiguous = True
        display_label = "happy_angry_suspect"
        display_label_ko = "행복? / 분노 의심"
        emotion_note = "입 다문 찡그림형 화난 표정일 수 있어요. 원본 확률은 그대로 표시합니다."
        guard_type = "closed_mouth_angry_like_happy"

    # 3) 하품/크게 입 벌림이 angry로 강하게 잡히는 케이스.
    #    모델에는 '하품' 클래스가 없으므로 확률은 그대로 두고 표시만 주의 문구로 바꿉니다.
    yawn_or_open_mouth_like = (
        top1_name == "angry"
        and angry >= 0.70
        and happy >= 0.06
        and relaxed <= 0.08
        and sad <= 0.08
    )
    if yawn_or_open_mouth_like:
        ambiguous = True
        display_label = "angry_yawn_suspect"
        display_label_ko = "분노? / 하품 의심"
        emotion_note = "입을 크게 벌린 하품/짖음 사진은 분노로 보일 수 있어요. 원본 확률은 그대로 표시합니다."
        guard_type = "open_mouth_yawn_like_angry"

    # 4) 근접 얼굴 + 입/이빨처럼 보이는 어두운 하단 영역인데 relaxed가 과확신되는 케이스.
    #    예: 흰 강아지 근접 얼굴, 입이 벌어졌는데 baseline이 편안 90% 이상으로 과확신.
    #    실제 확률은 그대로 두고 표시만 "편안? / 불편·분노 의심"으로 바꿉니다.
    visual_guard = relaxed_open_mouth_visual_suspect(img, raw_box)
    relaxed_open_mouth_like = (
        top1_name == "relaxed"
        and relaxed >= 0.90
        and visual_guard.get("suspect", False)
    )
    if relaxed_open_mouth_like:
        ambiguous = True
        display_label = "relaxed_uncomfortable_suspect"
        display_label_ko = "편안? / 불편·분노 의심"
        emotion_note = "근접 얼굴에서 입/이빨처럼 보이는 어두운 영역이 커서 편안으로 단정하지 않습니다. 원본 확률은 그대로 표시합니다."
        guard_type = "relaxed_open_mouth_like"

    return {
        "raw_probs_before_adjust": {CLASS_NAMES[i]: float(arr[i]) for i in range(len(CLASS_NAMES))},
        "postprocess": "display_guard_only",
        "angry_soft_boost": 0.0,
        "ambiguous": bool(ambiguous),
        "emotion_note": emotion_note,
        "guard_type": guard_type,
        "top2": top2_items,
        "display_label": display_label,
        "display_label_ko": display_label_ko,
        "display_confidence": display_confidence,
        "top2_margin": float(margin),
        "visual_guard": visual_guard if 'visual_guard' in locals() else {},
    }


# =========================================================
# Baseline / HuFEP / Ensemble emotion inference
# - YOLO 감지 코드는 그대로 유지
# - baseline 선택: baseline 체크포인트만 사용
# - HuFEP 선택: HuFEP 체크포인트만 사용
# - checkpoint가 CropFusion이면 5-crop -> ViT CLS -> crop_fusion 사용
# - checkpoint가 일반 timm 분류기이면 기존 multi-crop 평균 사용
# =========================================================
class CropTransformerFusion(nn.Module):
    """V15 학습 코드와 같은 5-crop Transformer Fusion Head.

    핵심:
    - 5개 crop의 ViT CLS feature를 사용
    - learnable CLS token 1개 + crop token 5개 = crop_pos [1, 6, hidden]
    - crop_weight_prior를 약하게 곱한 뒤 encoder -> norm -> dropout -> classifier
    - app 후처리/angry 보정 없음
    """
    def __init__(
        self,
        hidden_size: int,
        num_classes: int = 4,
        num_crops: int = 5,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.10,
        crop_weights: Optional[List[float]] = None,
        crop_weight_prior_strength: float = 0.15,
        crop_pos_len: Optional[int] = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            for h in [8, 6, 4, 3, 2, 1]:
                if hidden_size % h == 0:
                    num_heads = h
                    break
        self.num_crops = int(num_crops)
        self.hidden_size = int(hidden_size)
        self.crop_weight_prior_strength = float(crop_weight_prior_strength)

        # V15는 항상 CLS token + 5 crop token 구조입니다.
        # checkpoint의 crop_pos가 [1, 6, hidden]이면 그대로 맞춥니다.
        pos_len = int(crop_pos_len or (self.num_crops + 1))
        if pos_len < self.num_crops + 1:
            pos_len = self.num_crops + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.crop_pos = nn.Parameter(torch.zeros(1, pos_len, hidden_size))

        if crop_weights is None:
            crop_weights = [0.39, 0.28, 0.25, 0.05, 0.03]
        if len(crop_weights) != self.num_crops:
            crop_weights = [1.0 / max(1, self.num_crops)] * self.num_crops
        cw = torch.tensor(crop_weights, dtype=torch.float32).view(1, self.num_crops, 1)
        cw = cw / cw.sum().clamp_min(1e-8)
        self.register_buffer("crop_weight_prior", cw, persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.crop_pos, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, crop_cls_features: torch.Tensor) -> torch.Tensor:
        # crop_cls_features: [B, K, hidden]
        B, K, H = crop_cls_features.shape
        if self.crop_weight_prior_strength > 0:
            w = self.crop_weight_prior[:, :K, :].to(device=crop_cls_features.device, dtype=crop_cls_features.dtype)
            crop_cls_features = crop_cls_features * (1.0 + self.crop_weight_prior_strength * w)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, crop_cls_features], dim=1)  # [B, 1+K, hidden]
        x = x + self.crop_pos[:, :K + 1, :]
        x = self.encoder(x)
        pooled = x[:, 0]
        return self.classifier(self.dropout(self.norm(pooled)))

class HFCropFusionModel(nn.Module):
    def __init__(self, vit_config, hidden_size: int, num_classes: int = 4, num_crops: int = 5, crop_pos_len: Optional[int] = None):
        super().__init__()
        try:
            from transformers import ViTModel
        except Exception as e:
            raise RuntimeError("CropFusion HuFEP 모델은 transformers가 필요합니다. requirements.txt 설치를 다시 실행하세요.") from e
        self.vit = ViTModel(vit_config, add_pooling_layer=False)
        self.crop_fusion = CropTransformerFusion(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_crops=num_crops,
            num_layers=1,
            num_heads=4,
            dropout=0.10,
            crop_weights=[0.39, 0.28, 0.25, 0.05, 0.03],
            crop_weight_prior_strength=0.15,
            crop_pos_len=crop_pos_len,
        )

    def get_cls_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vit(pixel_values=pixel_values, interpolate_pos_encoding=True, return_dict=True)
        return out.last_hidden_state[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 5, 3, H, W]
        b, k, c, h, w = x.shape
        flat = x.view(b * k, c, h, w)
        cls = self.get_cls_features(flat).view(b, k, -1)
        return self.crop_fusion(cls)


def normalize_cropfusion_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = str(k)
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # 일부 학습 코드에서 이름이 달랐던 경우 보정
        if nk == "crop_fusion.pos_embed":
            nk = "crop_fusion.crop_pos"
        # 학습 코드마다 fusion CLS token 이름이 다를 수 있어서 app의 cls_token으로 통일
        if nk in {"crop_fusion.crop_cls", "crop_fusion.fusion_cls", "crop_fusion.cls", "crop_fusion.cls_embed"}:
            nk = "crop_fusion.cls_token"
        # V15 원본 구조는 crop_fusion.encoder / crop_fusion.classifier 입니다.
        # 이전 app처럼 classifier를 head로 바꾸면 classifier가 랜덤 초기화되어 확률이 25~33%로 퍼집니다.
        if nk.startswith("crop_fusion.head."):
            nk = nk.replace("crop_fusion.head.", "crop_fusion.classifier.", 1)
        if nk.startswith("crop_fusion.transformer."):
            nk = nk.replace("crop_fusion.transformer.", "crop_fusion.encoder.", 1)
        out[nk] = v
    return out


def is_cropfusion_checkpoint_state(state: Dict[str, Any]) -> bool:
    keys = list(state.keys())
    return any(k.startswith("vit.") for k in keys) and any(k.startswith("crop_fusion.") for k in keys)


def infer_vit_config_from_state(state: Dict[str, Any], classes: List[str]):
    try:
        from transformers import ViTConfig
    except Exception as e:
        raise RuntimeError("transformers가 설치되어 있지 않습니다. install_or_update.bat를 다시 실행하세요.") from e

    proj = state.get("vit.embeddings.patch_embeddings.projection.weight")
    if proj is None:
        raise RuntimeError("checkpoint에서 vit.embeddings.patch_embeddings.projection.weight를 찾지 못했습니다. HuFEP CropFusion 체크포인트인지 확인하세요.")
    hidden = int(proj.shape[0])
    channels = int(proj.shape[1])
    patch = int(proj.shape[2])
    pe = state.get("vit.embeddings.position_embeddings")
    image_size = int(os.environ.get("PET_AI_IMG_SIZE", "224"))
    if pe is not None and hasattr(pe, "shape"):
        n = int(pe.shape[1]) - 1
        side = int(round(math.sqrt(n)))
        if side * side == n:
            image_size = side * patch
    layer_ids = []
    for k in state.keys():
        if k.startswith("vit.encoder.layer."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                layer_ids.append(int(parts[3]))
    num_layers = max(layer_ids) + 1 if layer_ids else 12
    inter = state.get("vit.encoder.layer.0.intermediate.dense.weight")
    intermediate = int(inter.shape[0]) if inter is not None else hidden * 4
    heads = int(os.environ.get("PET_AI_HF_NUM_HEADS", "12" if hidden % 12 == 0 else "8"))
    if hidden == 1280 and "PET_AI_HF_NUM_HEADS" not in os.environ:
        heads = 16
    if hidden % heads != 0:
        heads = 8 if hidden % 8 == 0 else 4
    cfg = ViTConfig(
        image_size=image_size,
        patch_size=patch,
        num_channels=channels,
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        hidden_act="gelu",
        num_labels=len(classes),
    )
    return cfg, hidden, image_size


def load_cropfusion_engine(path: Path) -> Dict[str, Any]:
    cache_key = "cropfusion:" + str(path.resolve())
    if cache_key in ENGINE_CACHE:
        return ENGINE_CACHE[cache_key]
    ckpt = safe_torch_load(path)
    classes = checkpoint_classes(ckpt)
    raw_state = get_state_dict(ckpt)
    state = normalize_cropfusion_state(raw_state)
    if not is_cropfusion_checkpoint_state(state):
        raise RuntimeError(
            f"{path.name} 안에 crop_fusion/vit 구조가 없습니다. "
            "학습 때 저장한 HuFEP CropFusion 체크포인트를 넣어야 합니다."
        )
    cfg, hidden, img_size_val = infer_vit_config_from_state(state, classes)
    crop_pos = state.get("crop_fusion.crop_pos")
    crop_pos_len = int(crop_pos.shape[1]) if torch.is_tensor(crop_pos) and crop_pos.ndim == 3 else 5
    model = HFCropFusionModel(cfg, hidden_size=hidden, num_classes=len(classes), num_crops=5, crop_pos_len=crop_pos_len)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if k.startswith("crop_fusion.")]
    bad_unexpected = [k for k in unexpected if k.startswith("crop_fusion.")]
    if bad_missing or bad_unexpected:
        log("CropFusion load warning", path.name, "missing", bad_missing[:20], "unexpected", bad_unexpected[:20])
    critical_missing = [k for k in bad_missing if ("encoder" in k or "classifier" in k or "cls_token" in k or "crop_pos" in k)]
    if critical_missing:
        raise RuntimeError("CropFusion 핵심 weight가 로드되지 않았습니다: " + ", ".join(critical_missing[:12]))
    model = model.to(DEVICE).eval()
    tfm = T.Compose([
        T.Resize((img_size_val, img_size_val), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    engine = {
        "path": str(path),
        "name": path.name,
        "model": model,
        "classes": classes,
        "img_size": img_size_val,
        "transform": tfm,
        "timm_name": "HF-ViT-CropFusion",
        "strict": False,
        "cropfusion": True,
    }
    ENGINE_CACHE[cache_key] = engine
    log(f"Loaded CropFusion HuFEP {path.name} | img={img_size_val} | classes={classes} | hidden={hidden} | crop_pos_len={crop_pos_len}")
    return engine


def select_hufep_cropfusion_paths(mode: str) -> List[Path]:
    mode = normalize_mode(mode)
    paths = CHECKPOINTS[mode]["hufep"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("HuFEP 모델 파일이 없습니다: " + ", ".join(missing) + f" / 위치: {MODEL_DIR}")
    return paths


def make_cropfusion_5crops(img: Image.Image, raw_box: Optional[List[float]]) -> List[Tuple[str, Image.Image]]:
    w, h = img.size
    if raw_box is None:
        side = int(min(w, h) * 0.92)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        center = img.crop((x1, y1, x1 + side, y1 + side))
        return [
            ("crop_headsafe", center),
            ("crop_tight", center),
            ("crop_yolo", img.copy()),
            ("crop_center", center),
            ("crop_original", img.copy()),
        ]
    return [
        ("crop_headsafe", img.crop(face_safe_box(raw_box, w, h, pad=0.12))),
        ("crop_tight", img.crop(face_safe_box(raw_box, w, h, pad=0.04))),
        ("crop_yolo", img.crop(pad_box(raw_box, w, h, pad=0.12, square=True))),
        ("crop_center", img.crop(upper_head_box(raw_box, w, h, pad=0.08))),
        ("crop_original", img.copy()),
    ]


@torch.inference_mode()
def predict_cropfusion_5crops(img: Image.Image, raw_box: Optional[List[float]], engine: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    crops = make_cropfusion_5crops(img, raw_box)
    xs = [engine["transform"](cimg.convert("RGB")) for _, cimg in crops]
    x = torch.stack(xs, dim=0).unsqueeze(0).to(DEVICE)  # [1, 5, 3, H, W]
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        logits = engine["model"](x)
    raw = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
    return fixed_probs(raw, engine["classes"]), [name for name, _ in crops]

def make_emotion_crops(img: Image.Image, raw_box: Optional[List[float]], mode: str) -> List[Tuple[str, Image.Image, float]]:
    mode = normalize_mode(mode)
    w, h = img.size
    if raw_box is None:
        # fallback일 때는 전체 이미지 중심만 아주 약하게 분석
        side = int(min(w, h) * 0.92)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        return [("center", img.crop((x1, y1, x1 + side, y1 + side)), 1.0)]

    yolo_crop = img.crop(pad_box(raw_box, w, h, pad=0.12, square=True))
    upper_crop = img.crop(upper_head_box(raw_box, w, h, pad=0.10))
    face_crop = img.crop(face_safe_box(raw_box, w, h, pad=0.12))

    if mode == "fast":
        # 빠른모드: 1개 crop만. 분석 약하게/빠르게.
        return [("upper_head", upper_crop, 1.0)]
    # 예비 경로: 현재 공용 서버에서는 normalize_mode()가 항상 fast를 반환하므로 사용되지 않습니다.
    return [("face_safe", face_crop, 0.50), ("upper_head", upper_crop, 0.30), ("yolo", yolo_crop, 0.20)]


def checkpoint_group(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("baseline") or "baseline" in name:
        return "baseline"
    if name.startswith("hufep") or "hufep" in name:
        return "hufep"
    return "unknown"


def load_any_emotion_engine(path: Path) -> Dict[str, Any]:
    """checkpoint 구조를 확인해서 CropFusion 또는 일반 timm 모델로 로드합니다."""
    ckpt = safe_torch_load(path)
    state = normalize_cropfusion_state(get_state_dict(ckpt))
    if is_cropfusion_checkpoint_state(state):
        engine = load_cropfusion_engine(path)
        engine["engine_type"] = "cropfusion"
        return engine
    engine = load_engine(path)
    engine["engine_type"] = "timm"
    return engine


@torch.inference_mode()
def predict_timm_multicrop(img: Image.Image, raw_box: Optional[List[float]], mode: str, engine: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    crops = make_emotion_crops(img, raw_box, mode)
    total = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    wsum = 0.0
    names: List[str] = []
    for cname, cimg, weight in crops:
        prob = predict_one_crop(cimg, engine)
        total += prob * float(weight)
        wsum += float(weight)
        names.append(cname)
    if wsum > 0:
        total = total / wsum
    total = total / max(1e-9, float(total.sum()))
    return total.astype(np.float32), names


def build_group_summary(model_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[np.ndarray]] = {}
    for part in model_parts:
        arr = np.asarray(part.get("probs_array"), dtype=np.float32)
        if arr.shape[0] != len(CLASS_NAMES):
            continue
        groups.setdefault(str(part.get("group", "unknown")), []).append(arr)
    out: Dict[str, Any] = {}
    for group, arrs in groups.items():
        avg = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)
        avg = avg / max(1e-9, float(avg.sum()))
        idx = int(np.argmax(avg))
        out[group] = {
            "label": CLASS_NAMES[idx],
            "label_ko": CLASS_KO.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
            "confidence": float(avg[idx]),
            "probs": {CLASS_NAMES[i]: float(avg[i]) for i in range(len(CLASS_NAMES))},
            "checkpoints": [p.get("checkpoint") for p in model_parts if p.get("group") == group],
        }
    return out


@torch.inference_mode()
def classify_dog(img: Image.Image, raw_box: Optional[List[float]], mode: str, model_choice: str) -> Dict[str, Any]:
    """모델 선택을 실제로 반영하는 감정 분류.

    - baseline 선택: baseline_best_* 체크포인트만 분석
    - HuFEP 선택: hufep_best_* 체크포인트만 분석
    - 각 체크포인트가 CropFusion 구조이면 5-crop CropFusion으로 실행
    - 일반 timm 구조이면 기존 multi-crop 평균으로 실행
    """
    mode = normalize_mode(mode)
    requested = normalize_model(model_choice)
    paths = select_model_paths(mode, requested)

    model_probs: List[np.ndarray] = []
    model_parts: List[Dict[str, Any]] = []
    crop_names_used: List[str] = []

    for p in paths:
        group = checkpoint_group(p)
        engine = load_any_emotion_engine(p)
        if engine.get("engine_type") == "cropfusion" or engine.get("cropfusion"):
            prob, crop_names = predict_cropfusion_5crops(img, raw_box, engine)
            engine_type = "CropFusion"
        else:
            prob, crop_names = predict_timm_multicrop(img, raw_box, mode, engine)
            engine_type = "timm-multicrop"

        prob = prob.astype(np.float32)
        prob = prob / max(1e-9, float(prob.sum()))
        model_probs.append(prob)
        crop_names_used = crop_names
        model_parts.append({
            "group": group,
            "checkpoint": p.name,
            "crops": crop_names,
            "img_size": engine.get("img_size"),
            "timm": engine.get("timm_name"),
            "engine_type": engine_type,
            "cropfusion": bool(engine_type == "CropFusion"),
            "probs": {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))},
            "probs_array": prob,
        })

    if not model_probs:
        raise RuntimeError("사용할 감정 체크포인트가 없습니다.")

    probs = np.mean(np.stack(model_probs, axis=0), axis=0).astype(np.float32)
    probs = probs / max(1e-9, float(probs.sum()))
    idx = int(np.argmax(probs))
    post = fair_identity_emotion_meta(probs, img=img, raw_box=raw_box)

    # JSON 응답에서는 numpy array를 제거합니다.
    model_parts_json = []
    for part in model_parts:
        cp = dict(part)
        cp.pop("probs_array", None)
        model_parts_json.append(cp)

    return {
        "label": CLASS_NAMES[idx],
        "label_ko": CLASS_KO.get(CLASS_NAMES[idx], CLASS_NAMES[idx]),
        "confidence": float(probs[idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "model_choice": requested,
        "requested_model_choice": requested,
        "actual_model_choice": requested,
        "mode": mode,
        "checkpoints": [p.name for p in paths],
        "model_parts": model_parts_json,
        "model_group_summary": build_group_summary(model_parts),
        "crop_names": crop_names_used,
        **post,
    }



def draw_results(img: Image.Image, results: List[Dict[str, Any]]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("malgun.ttf", 24)
        small_font = ImageFont.truetype("malgun.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        text = f"Dog {r['dog_id']} | {r.get('display_label_ko', r['label_ko'])} {r.get('display_confidence', r['confidence'])*100:.1f}%"
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 260, 28
        y_text = max(0, y1 - th - 10)
        draw.rectangle([x1, y_text, min(out.width, x1 + tw + 12), y_text + th + 10], fill=(0, 0, 0))
        draw.text((x1 + 6, y_text + 4), text, fill=(0, 255, 0), font=font)
        if r.get("fallback"):
            draw.text((x1 + 6, min(out.height - 20, y2 + 6)), "YOLO 미검출 → 전체 이미지 fallback", fill=(255, 180, 0), font=small_font)
    return out


def analyze_image_pil(img: Image.Image, mode: str, model_choice: str, save_prefix: str = "img") -> Dict[str, Any]:
    mode = normalize_mode(mode)
    cfg = MODE_CONFIGS[mode]
    detections = detect_dogs(img, mode)
    w, h = img.size
    results = []
    if not detections:
        pred = classify_dog(img, raw_box=None, mode=mode, model_choice=model_choice)
        results.append({
            "dog_id": 1,
            "box": [5, 5, max(6, w - 5), max(6, h - 5)],
            "raw_box": None,
            "det_conf": None,
            "fallback": True,
            "yolo": cfg["yolo_name"],
            **pred,
        })
    else:
        for i, d in enumerate(detections, 1):
            pred = classify_dog(img, raw_box=d["raw_box"], mode=mode, model_choice=model_choice)
            results.append({
                "dog_id": i,
                "box": d["box"],
                "raw_box": d["raw_box"],
                "det_conf": d["det_conf"],
                "fallback": False,
                "yolo": d["yolo"],
                "imgsz": d["imgsz"],
                **pred,
            })
    annotated = draw_results(img, results)
    uid = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{save_prefix}_{uid}.jpg"
    annotated.save(out_path, quality=92)
    return {
        "ok": True,
        "mode": mode,
        "mode_title": cfg["title"],
        "model_choice": normalize_model(model_choice),
        "model_title": MODEL_DISPLAY_NAMES.get(normalize_model(model_choice), normalize_model(model_choice)),
        "yolo": cfg["yolo_name"],
        "num_dogs": len(results),
        "source_width": w,
        "source_height": h,
        "annotated_url": f"/outputs/{out_path.name}",
        "results": results,
        "detections": results,
    }


def save_upload(upload: UploadFile, prefix: str) -> Path:
    ext = Path(upload.filename or "file").suffix or ".bin"
    path = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def convert_mp4_for_browser(raw_path: Path, final_path: Path) -> Path:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", str(raw_path), "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-an", str(final_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and final_path.exists() and final_path.stat().st_size > 0:
            return final_path
        log("ffmpeg 변환 실패:", proc.stderr[-1000:])
    except Exception as e:
        log("ffmpeg 변환 생략:", repr(e))
    return raw_path


def analyze_video_file(path: Path, mode: str, model_choice: str) -> Dict[str, Any]:
    mode = normalize_mode(mode)
    cfg = MODE_CONFIGS[mode]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("동영상을 열 수 없습니다.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(cfg["video_stride"]))
    uid = uuid.uuid4().hex[:12]
    raw_out = OUTPUT_DIR / f"video_{uid}_raw.mp4"
    final_out = OUTPUT_DIR / f"video_{uid}.mp4"
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
    last_results: Optional[List[Dict[str, Any]]] = None
    summary_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    processed = 0
    t0 = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0 or last_results is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data = analyze_image_pil(pil, mode, model_choice, save_prefix="video_frame")
            last_results = data["results"]
            for r in last_results:
                summary_counts[r["label"]] = summary_counts.get(r["label"], 0) + 1
            processed += 1
        if last_results:
            pil_annot = draw_results(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), last_results)
            frame = cv2.cvtColor(np.array(pil_annot), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    playable = convert_mp4_for_browser(raw_out, final_out)
    elapsed = time.time() - t0
    return {
        "ok": True,
        "mode": mode,
        "mode_title": cfg["title"],
        "model_choice": normalize_model(model_choice),
        "model_title": MODEL_DISPLAY_NAMES.get(normalize_model(model_choice), normalize_model(model_choice)),
        "yolo": cfg["yolo_name"],
        "video_url": f"/outputs/{playable.name}",
        "frames": frame_idx,
        "total_frames_reported": total_frames,
        "processed_frames": processed,
        "stride": stride,
        "elapsed_sec": elapsed,
        "summary_counts": summary_counts,
        "last_results": last_results or [],
    }


# =========================================================
# 1. FastAPI
# =========================================================
app = FastAPI(title="Pet-AI Dog Emotion", version="public-yolo26n-lite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health")
def health():
    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "model_dir": str(MODEL_DIR),
        "device": DEVICE,
        "modes": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k, v in MODE_CONFIGS.items()},
        "checkpoint_files": {m: {c: [p.name for p in ps] for c, ps in d.items()} for m, d in CHECKPOINTS.items()},
        "exists": {
            p.name: p.exists()
            for group in CHECKPOINTS.values()
            for ps in group.values()
            for p in ps
        } | {"yolo26n.pt": FAST_YOLO_PATH.exists()},
    }


# =========================================================
# 2. UI: app-4.py 스타일 단계별 선택 화면
# =========================================================
@app.get("/", response_class=HTMLResponse)
def index():
    # 공용 서버 버전은 첫 화면에서 바로 모델을 선택합니다.
    return select_model("fast")


@app.get("/select-model", response_class=HTMLResponse)
def select_model(mode: str = Query("fast")):
    mode = normalize_mode(mode)
    mode_title = MODE_CONFIGS[mode]["title"]
    return f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>모델 선택</title>
  <style>
    :root {{ --main:#1f8b78; --bg:#f5f7f9; --card:#ffffff; --text:#111827; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:var(--bg); color:var(--text); }}
    .topbar {{ height:12px; background:var(--main); }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 40px 18px 60px; }}
    h1 {{ display:inline-block; margin:0 0 10px; padding:4px 8px; font-size:42px; background:#d1d5db; letter-spacing:-1px; }}
    .sub {{ color:#6b7280; margin:0 0 18px; font-size:22px; }}
    .chip {{ display:inline-block; padding:8px 12px; border-radius:999px; background:#ecfdf5; color:#047857; font-weight:800; margin-bottom:18px; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap:20px; margin-top:24px; }}
    .card {{ border:0; border-radius:22px; background:white; box-shadow:0 8px 24px rgba(0,0,0,.08); padding:28px; text-align:left; cursor:pointer; min-height:210px; }}
    .card:hover {{ transform: translateY(-2px); box-shadow:0 12px 30px rgba(0,0,0,.12); }}
    .title {{ font-size:24px; font-weight:900; margin-bottom:10px; }}
    .desc {{ color:#4b5563; line-height:1.65; font-size:15px; }}
    .small {{ color:#6b7280; margin-top:28px; line-height:1.7; }}
    .caution {{ margin-top:22px; padding:14px 16px; border-radius:14px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; line-height:1.65; font-weight:700; }}
    .backbtn {{ display:inline-block; padding:10px 14px; border-radius:999px; background:#111827; color:white; text-decoration:none; font-weight:900; margin-bottom:18px; }}
    @media (max-width:900px){{ .grid{{grid-template-columns:1fr;}} h1{{font-size:34px;}} .wrap{{padding:28px 12px;}} }}
  </style>
</head>
<body>
<div class="topbar"></div>
<div class="wrap">
  <a class="backbtn" href="/">← 이전으로</a>
  <h1>반려견 감정 분석</h1>
  <p class="sub">감정 모델을 선택하세요</p>
  <div class="chip">YOLO26n / IMG_SIZE 224 / 최대 3마리</div>
  <div class="grid">
    <button class="card" onclick="location.href='/ui?mode={mode}&model=baseline'">
      <div class="title">baseline</div>
      <div class="desc">강아지 감정 데이터 기준 baseline 모델입니다. 단독 모델 비교용입니다.</div>
    </button>
    <button class="card" onclick="location.href='/ui?mode={mode}&model=hufep'">
      <div class="title">HuFEP</div>
      <div class="desc">HuFEP 기반 모델입니다. baseline과 비교해서 확인하세요.</div>
    </button>
  </div>
  <div class="caution">주의: 감정 분석은 100% 정확하지 않습니다. baseline과 HuFEP도 사진 조건에 따라 서로 다른 판단을 할 수 있고, 하품/입벌림/흐림/겹침 이미지에서는 실수가 많이 날 수 있습니다.</div>
  <p class="small"><a href="/health" target="_blank">/health 확인</a></p>
</div>
</body>
</html>
    """


@app.get("/ui", response_class=HTMLResponse)
def ui(mode: str = Query("fast"), model: str = Query("hufep")):
    mode = normalize_mode(mode)
    model = normalize_model_choice(model)
    mode_title = MODE_CONFIGS[mode]["title"]
    model_title = MODEL_DISPLAY_NAMES[model]
    html = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>반려견 감정 분석 - __MODEL_TITLE__</title>
  <style>
    :root { --main:#1f8b78; --bg:#f5f7f9; --card:#ffffff; --text:#111827; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, 'Malgun Gothic', sans-serif; background:var(--bg); color:var(--text); }
    .topbar { height:12px; background:var(--main); }
    .wrap { max-width: 1120px; margin: 0 auto; padding: 40px 18px 60px; }
    h1 { display:inline-block; margin: 0 0 6px; padding: 4px 8px; font-size: 42px; background:#d1d5db; letter-spacing:-1px; }
    .sub { color:#6b7280; margin:0 0 18px; font-size:22px; }
    .navback { display:inline-block; padding:10px 14px; border-radius:999px; background:#111827; color:white; text-decoration:none; font-weight:900; margin:8px 0 14px; }
    .chips { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:20px; }
    .chip { display:inline-block; padding:8px 13px; border-radius:999px; background:#ecfdf5; color:#047857; font-weight:900; }
    .chip.blue { background:#eff6ff; color:#1d4ed8; }
    .menu { display:grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap:22px; margin: 28px 0; }
    .menu button, .action, .back { border:0; border-radius:18px; padding:22px; background:white; box-shadow:0 6px 18px rgba(0,0,0,.07); cursor:pointer; font-weight:900; font-size:18px; }
    .menu button { min-height:80px; }
    .panel { display:none; background:var(--card); border-radius:18px; padding:18px; box-shadow:0 2px 14px rgba(0,0,0,.08); margin-top:16px; }
    .panel.active { display:block; }
    .back { background:#111827; color:white; padding:10px 14px; margin-bottom:10px; font-size:14px; }
    .action { background:var(--main); color:white; padding:12px 16px; margin:10px 8px 10px 0; font-size:15px; }
    .action.secondary { background:#374151; }
    .action.danger { background:#b91c1c; }
    input[type=file] { display:block; margin:12px 0; padding:12px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; width:100%; }
    .result { margin-top:16px; }
    .status { color:#374151; white-space:pre-line; margin:10px 0; }
    .warn { color:#b91c1c; font-weight:700; white-space:pre-line; }
    .ok { color:#047857; font-weight:700; white-space:pre-line; }
    img, video { max-width:100%; border-radius:14px; margin-top:12px; background:#000; }
    .dog { background:#f9fafb; border:1px solid #e5e7eb; border-radius:14px; padding:12px; margin-top:10px; }
    .small { color:#6b7280; font-size:13px; line-height:1.55; }
    .caution { margin-top:14px; padding:14px 16px; border-radius:14px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; line-height:1.65; font-weight:700; }
    .cameraRow { display:grid; grid-template-columns: 1fr; gap:12px; }
    #photoVideo { width:100%; max-width:720px; min-height:240px; background:#111; }
    .liveBox { position:relative; width:100%; max-width:720px; background:#111; border-radius:14px; overflow:hidden; margin-top:12px; }
    #webcamVideo { width:100%; display:block; margin-top:0; border-radius:14px; background:#111; }
    #webcamOverlay { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
    .liveInfo { margin-top:10px; padding:10px 12px; background:#ecfdf5; border:1px solid #a7f3d0; border-radius:12px; color:#065f46; font-weight:700; }
    @media (max-width:720px){ .menu{grid-template-columns:1fr;} h1{font-size:34px;} .wrap{padding:24px 12px;} }
  </style>
</head>
<body>
<div class="topbar"></div>
<div class="wrap">
  <h1>반려견 감정 분석</h1>
  <p class="sub">선택한 모델로 최대 3마리까지 분석</p>
  <a class="navback" href="/select-model?mode=__APP_MODE__">← 이전으로</a>
  <div class="chips">
    <div class="chip blue">현재 모델: __MODEL_TITLE__</div>
  </div>
  <div class="caution">주의: 이 앱은 100% 정확하지 않습니다. 강아지 자세, 조명, 흔들림, 흰/검은 털, 하품, 입 벌림, 여러 마리 겹침 때문에 감정과 마릿수를 헷갈릴 수 있습니다.</div>

  <div id="home" class="menu">
    <button onclick="openPanel('imagePanel')">이미지 선택</button>
    <button onclick="openPanel('videoPanel')">동영상 선택</button>
    <button onclick="openPanel('photoPanel')">사진 찍기</button>
    <button onclick="openPanel('webcamPanel')">웹캠 실시간</button>
  </div>

  <div id="imagePanel" class="panel">
    <button class="back" onclick="goHome()">← 이전으로</button>
    <h2>이미지 선택</h2>
    <input id="imageInput" type="file" accept="image/*" />
    <button class="action" onclick="sendImage('imageInput','imageResult')">이미지 분석</button>
    <div id="imageResult" class="result"></div>
  </div>

  <div id="videoPanel" class="panel">
    <button class="back" onclick="goHome()">← 이전으로</button>
    <h2>동영상 선택</h2>
    <p class="small">공용 서버 lite 버전은 YOLO26n만 사용합니다. 동영상은 서버 사양에 따라 오래 걸릴 수 있습니다.</p>
    <input id="videoInput" type="file" accept="video/*" />
    <button class="action" onclick="sendVideo()">동영상 분석</button>
    <div id="videoResult" class="result"></div>
  </div>

  <div id="photoPanel" class="panel">
    <button class="back" onclick="goHome()">← 이전으로</button>
    <h2>사진 찍기</h2>
    <button class="action" onclick="startPhotoCamera()">카메라 켜기</button>
    <button class="action secondary" onclick="switchPhotoCamera()">전면/후면 전환</button>
    <button class="action secondary" onclick="capturePhotoAndAnalyze()">사진 찍고 분석</button>
    <button class="action danger" onclick="stopPhotoCamera()">카메라 끄기</button>
    <div class="cameraRow"><video id="photoVideo" autoplay playsinline muted></video><canvas id="photoCanvas" style="display:none"></canvas></div>
    <div id="photoResult" class="result"></div>
  </div>

  <div id="webcamPanel" class="panel">
    <button class="back" onclick="goHome()">← 이전으로</button>
    <h2>웹캠 실시간 분석</h2>
    <button class="action" onclick="startWebcam()">웹캠 켜기</button>
    <button class="action secondary" onclick="switchWebcamCamera()">전면/후면 전환</button>
    <button class="action secondary" onclick="startLiveAnalysis()">실시간 분석 시작</button>
    <button class="action secondary" onclick="captureWebcamOnce()">현재 화면 1회 분석</button>
    <button class="action danger" onclick="stopWebcam()">웹캠 끄기</button>
    <div class="liveBox"><video id="webcamVideo" autoplay playsinline muted></video><canvas id="webcamOverlay"></canvas></div>
    <canvas id="webcamCanvas" style="display:none"></canvas>
    <div id="webcamResult" class="result"></div>
  </div>

  <p class="small"><a href="/" target="_self">모델 선택으로 돌아가기</a> · <a href="/health" target="_blank">/health 확인</a></p>
</div>

<script>
const APP_MODE='__APP_MODE__';
const APP_MODEL='__APP_MODEL__';
let webcamStream=null, photoStream=null, liveRunning=false, liveBusy=false, liveTimer=null;
let photoFacingMode='environment', webcamFacingMode='environment';
function api(url){ const sep = url.includes('?') ? '&' : '?'; return url + sep + 'mode=' + encodeURIComponent(APP_MODE) + '&model=' + encodeURIComponent(APP_MODEL); }
function facingLabel(mode){ return mode === 'user' ? '전면 카메라' : '후면 카메라'; }
function toggleFacing(mode){ return mode === 'user' ? 'environment' : 'user'; }
function cameraConstraints(mode, purpose){ const isPhoto=purpose==='photo'; return {video:{facingMode:{ideal:mode}, width:{ideal:isPhoto?1280:640}, height:{ideal:isPhoto?720:480}, frameRate:{ideal:isPhoto?30:15,max:isPhoto?30:20}}, audio:false}; }
function openPanel(id){ document.getElementById('home').style.display='none'; document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active')); document.getElementById(id).classList.add('active'); }
function goHome(){ stopWebcam(); stopPhotoCamera(); document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active')); document.getElementById('home').style.display='grid'; }
function escapeHtml(s){ return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function renderDogs(dogs){ if(!dogs||!dogs.length)return '<p>결과 없음</p>'; return dogs.map(d=>{ const label=d.display_label||d.display_label_ko||d.label_ko||d.label||'?'; const conf=d.display_confidence!=null?(d.display_confidence*100).toFixed(1)+'%':(d.confidence!=null?(d.confidence*100).toFixed(1)+'%':''); const yoloName=d.yolo||d.detector||''; const det=d.det_conf!=null?(' / YOLO: '+escapeHtml(yoloName||'detected')+' conf '+Number(d.det_conf).toFixed(3)):(d.fallback?' / YOLO 미검출 fallback':''); const probs=d.probs?Object.entries(d.probs).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join('<br>'):''; const note=d.emotion_note?`<div class="note">※ ${escapeHtml(d.emotion_note)}</div>`:''; return `<div class="dog"><b>Dog ${d.dog_id??''}: ${escapeHtml(label)} ${conf}${det}</b><br><span class="small">fallback: ${!!d.fallback} / 모델: ${escapeHtml(d.model_choice||APP_MODEL)} / 체크포인트: ${escapeHtml((d.checkpoints||[]).join(', '))}</span>${note}${probs?'<div class="small">'+probs+'</div>':''}</div>`; }).join(''); }
async function postFile(url,file){ const fd=new FormData(); fd.append('file',file); const res=await fetch(api(url),{method:'POST',body:fd}); let data=null; try{data=await res.json();}catch(e){} if(!res.ok) throw new Error((data&&(data.detail||data.error))||('HTTP '+res.status)); return data; }
function renderImageResult(box,data){ const url=data.result_image_url||data.annotated_url; const fallbackMsg=(data.results||[]).some(d=>d.fallback)?'<div class="warn">YOLO가 강아지를 못 찾은 항목은 전체 이미지 fallback으로 분석했습니다.</div>':''; box.innerHTML=`<div class="ok">분석 완료 / 모델: ${escapeHtml(data.model_title||APP_MODEL)} / 검출: ${data.num_dogs??(data.results||[]).length}마리</div>`+fallbackMsg+(url?`<img src="${url}?t=${Date.now()}">`:'')+renderDogs(data.detections||data.results||[]); }
async function sendImage(inputId,resultId){ const input=document.getElementById(inputId), box=document.getElementById(resultId); if(!input.files||!input.files[0]){box.innerHTML='<div class="warn">파일을 선택하세요.</div>';return;} box.innerHTML='<div class="status">분석 중입니다...</div>'; try{const data=await postFile('/predict/image',input.files[0]); renderImageResult(box,data);}catch(e){box.innerHTML='<div class="warn">분석 실패: '+escapeHtml(e.message)+'</div>';} }
async function sendVideo(){ const input=document.getElementById('videoInput'), box=document.getElementById('videoResult'); if(!input.files||!input.files[0]){box.innerHTML='<div class="warn">동영상을 선택하세요.</div>';return;} const qs='frame_stride=20&max_frames=150'; box.innerHTML='<div class="status">동영상 분석 중입니다...</div>'; try{const data=await postFile('/predict/video?'+qs,input.files[0]); const vurl=data.result_video_url||data.video_url; const lurl=data.log_url; box.innerHTML=`<div class="ok">동영상 분석 완료 / 처리 프레임: ${data.processed_frames??''} / 모델: ${escapeHtml(data.model_title||APP_MODEL)}</div>`+(vurl?`<video src="${vurl}?t=${Date.now()}" controls preload="metadata"></video><br><a href="${vurl}" target="_blank">결과 영상 새 창으로 열기</a>`:'')+(lurl?` · <a href="${lurl}" target="_blank">로그 열기</a>`:'');}catch(e){box.innerHTML='<div class="warn">분석 실패: '+escapeHtml(e.message)+'</div>';} }
async function startPhotoCamera(){ const box=document.getElementById('photoResult'); try{stopPhotoCamera(false); photoStream=await navigator.mediaDevices.getUserMedia(cameraConstraints(photoFacingMode,'photo')); document.getElementById('photoVideo').srcObject=photoStream; box.innerHTML='<div class="ok">카메라 연결 성공: '+facingLabel(photoFacingMode)+'</div>';}catch(e){try{photoStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); document.getElementById('photoVideo').srcObject=photoStream; box.innerHTML='<div class="ok">카메라 연결 성공.</div>';}catch(e2){box.innerHTML='<div class="warn">카메라 연결 실패: '+escapeHtml(e2.message)+'</div>';}} }
async function switchPhotoCamera(){ photoFacingMode=toggleFacing(photoFacingMode); await startPhotoCamera(); }
function stopPhotoCamera(){ if(photoStream){photoStream.getTracks().forEach(t=>t.stop()); photoStream=null;} const v=document.getElementById('photoVideo'); if(v)v.srcObject=null; }
async function capturePhotoAndAnalyze(){ const video=document.getElementById('photoVideo'), canvas=document.getElementById('photoCanvas'), box=document.getElementById('photoResult'); if(!video.videoWidth){box.innerHTML='<div class="warn">먼저 카메라를 켜세요.</div>';return;} canvas.width=video.videoWidth; canvas.height=video.videoHeight; canvas.getContext('2d').drawImage(video,0,0); box.innerHTML='<div class="status">촬영 사진 분석 중...</div>'; canvas.toBlob(async blob=>{ try{const file=new File([blob],'photo.jpg',{type:'image/jpeg'}); const data=await postFile('/predict/image',file); renderImageResult(box,data);}catch(e){box.innerHTML='<div class="warn">분석 실패: '+escapeHtml(e.message)+'</div>';} },'image/jpeg',.92); }
function resizeOverlayCanvas(){ const v=document.getElementById('webcamVideo'), o=document.getElementById('webcamOverlay'); if(!v||!o||!v.videoWidth)return; o.width=v.videoWidth; o.height=v.videoHeight; }
function clearOverlay(){ const o=document.getElementById('webcamOverlay'); if(!o)return; o.getContext('2d').clearRect(0,0,o.width,o.height); }
function drawLiveOverlay(data){ const v=document.getElementById('webcamVideo'), o=document.getElementById('webcamOverlay'); if(!v||!o||!v.videoWidth)return; resizeOverlayCanvas(); const ctx=o.getContext('2d'); ctx.clearRect(0,0,o.width,o.height); const dogs=(data&&(data.detections||data.results))||[]; const sx=o.width/Math.max(1,Number(data.source_width)||o.width); const sy=o.height/Math.max(1,Number(data.source_height)||o.height); ctx.lineWidth=Math.max(3,Math.round(o.width/360)); ctx.font=`${Math.max(18,Math.round(o.width/34))}px Arial`; ctx.textBaseline='top'; dogs.forEach((d,idx)=>{ if(!d.box)return; let [x1,y1,x2,y2]=d.box.map(Number); x1*=sx;x2*=sx;y1*=sy;y2*=sy; const label=d.display_label||d.label||'?'; const title=`Dog${d.dog_id??idx+1}: ${label} ${d.confidence!=null?Number(d.confidence).toFixed(2):''}`; ctx.strokeStyle='#00ff55'; ctx.strokeRect(x1,y1,Math.max(1,x2-x1),Math.max(1,y2-y1)); const m=ctx.measureText(title), th=Math.max(24,Math.round(o.width/30)), ty=Math.max(0,y1-th-4); ctx.fillStyle='rgba(0,0,0,.82)'; ctx.fillRect(x1,ty,m.width+14,th+6); ctx.fillStyle='#00ff55'; ctx.fillText(title,x1+7,ty+5); }); }
function renderLiveCards(box,data,msg='실시간 분석 갱신'){ const dogs=data.detections||data.results||[]; box.innerHTML=`<div class="liveInfo">${msg} / 모델: ${escapeHtml(data.model_title||APP_MODEL)} / 검출: ${data.num_dogs??dogs.length}마리</div>`+renderDogs(dogs); drawLiveOverlay(data); }
async function startWebcam(){ const box=document.getElementById('webcamResult'); try{ if(!webcamStream){webcamStream=await navigator.mediaDevices.getUserMedia(cameraConstraints(webcamFacingMode,'webcam'));} const v=document.getElementById('webcamVideo'); v.srcObject=webcamStream; v.onloadedmetadata=()=>{resizeOverlayCanvas();clearOverlay();}; box.innerHTML='<div class="ok">카메라 연결 성공: '+facingLabel(webcamFacingMode)+'</div>';}catch(e){try{webcamStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:15,max:20}},audio:false}); const v=document.getElementById('webcamVideo'); v.srcObject=webcamStream; v.onloadedmetadata=()=>{resizeOverlayCanvas();clearOverlay();}; box.innerHTML='<div class="ok">카메라 연결 성공.</div>';}catch(e2){box.innerHTML='<div class="warn">웹캠 연결 실패: '+escapeHtml(e2.message)+'</div>';}} }
async function switchWebcamCamera(){ const was=liveRunning; liveRunning=false; liveBusy=false; if(liveTimer){clearTimeout(liveTimer);liveTimer=null;} if(webcamStream){webcamStream.getTracks().forEach(t=>t.stop()); webcamStream=null;} clearOverlay(); webcamFacingMode=toggleFacing(webcamFacingMode); await startWebcam(); if(was)startLiveAnalysis(); }
function stopWebcam(){ liveRunning=false; liveBusy=false; if(liveTimer){clearTimeout(liveTimer);liveTimer=null;} if(webcamStream){webcamStream.getTracks().forEach(t=>t.stop()); webcamStream=null;} const v=document.getElementById('webcamVideo'); if(v)v.srcObject=null; clearOverlay(); }
async function captureWebcamBlob(maxWidth=384,quality=.55){ const v=document.getElementById('webcamVideo'), c=document.getElementById('webcamCanvas'); if(!v.videoWidth)throw new Error('먼저 웹캠을 켜세요.'); const scale=Math.min(1,maxWidth/v.videoWidth); c.width=Math.max(1,Math.round(v.videoWidth*scale)); c.height=Math.max(1,Math.round(v.videoHeight*scale)); c.getContext('2d').drawImage(v,0,0,c.width,c.height); return new Promise(resolve=>c.toBlob(blob=>resolve(blob),'image/jpeg',quality)); }
async function captureWebcamOnce(){ const box=document.getElementById('webcamResult'); box.innerHTML='<div class="status">현재 화면 분석 중...</div>'; try{const blob=await captureWebcamBlob(384,.58); const file=new File([blob],'webcam.jpg',{type:'image/jpeg'}); const data=await postFile('/predict/webcam',file); renderLiveCards(box,data,'현재 화면 1회 분석 완료');}catch(e){box.innerHTML='<div class="warn">분석 실패: '+escapeHtml(e.message)+'</div>';} }
function startLiveAnalysis(){ const box=document.getElementById('webcamResult'); if(!webcamStream){box.innerHTML='<div class="warn">먼저 웹캠을 켜세요.</div>';return;} if(liveRunning){return;} liveRunning=true; box.innerHTML='<div class="status">실시간 분석 시작...</div>'; liveLoop(); }
async function liveLoop(){ if(!liveRunning)return; if(liveBusy){liveTimer=setTimeout(liveLoop,150);return;} liveBusy=true; const box=document.getElementById('webcamResult'); try{const blob=await captureWebcamBlob(384,.55); const file=new File([blob],'webcam_live.jpg',{type:'image/jpeg'}); const data=await postFile('/predict/webcam',file); renderLiveCards(box,data,'실시간 분석 갱신');}catch(e){box.innerHTML='<div class="warn">실시간 분석 실패: '+escapeHtml(e.message)+'</div>';} liveBusy=false; liveTimer=setTimeout(liveLoop, 350); }
</script>
</body>
</html>
    """
    return html.replace("__APP_MODE__", mode).replace("__APP_MODEL__", model).replace("__MODE_TITLE__", mode_title).replace("__MODEL_TITLE__", model_title)



# =========================================================
# 3. API: app(36).py 강아지 마릿수/감정분석 그대로 사용
# =========================================================
@app.post("/api/analyze-image")
def api_analyze_image(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")
    in_path = save_upload(file, "image")
    try:
        img = Image.open(in_path).convert("RGB")
        return JSONResponse(analyze_image_pil(img, mode, model, save_prefix="image"))
    except Exception as e:
        log("image error", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-frame")
def api_analyze_frame(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")
    in_path = save_upload(file, "frame")
    try:
        img = Image.open(in_path).convert("RGB")
        return JSONResponse(analyze_image_pil(img, mode, model, save_prefix="frame"))
    except Exception as e:
        log("frame error", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-video")
def api_analyze_video(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    if not file.content_type or not file.content_type.startswith("video/"):
        # 일부 브라우저가 octet-stream으로 보낼 수도 있음. 확장자로 보완.
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise HTTPException(status_code=400, detail="동영상 파일이 필요합니다.")
    in_path = save_upload(file, "video")
    try:
        return JSONResponse(analyze_video_file(in_path, mode, model))
    except Exception as e:
        log("video error", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


# 기존 이름 호환용
@app.post("/predict/image")
def predict_image_compat(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    return api_analyze_image(file, mode, model)

@app.post("/predict/webcam")
def predict_webcam_compat(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    return api_analyze_frame(file, mode, model)

@app.post("/predict/video")
def predict_video_compat(file: UploadFile = File(...), mode: str = Query("fast"), model: str = Query("hufep")):
    return api_analyze_video(file, mode, model)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
