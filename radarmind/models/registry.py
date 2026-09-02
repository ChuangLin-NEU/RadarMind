"""ModelRegistry. Inference only — do not import Phase 1 training loops."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from radarmind.data.base import RadarSample
from radarmind.data.ci4r_adapter import CLASS_NAMES as CI4R_CLASSES
from radarmind.data.mri_adapter import CLASS_NAMES as MRI_CLASSES
from radarmind.data.radhar_adapter import CLASS_NAMES as RAD_CLASSES
from radarmind.models.openset import apply_open_set, load_app_pack
from radarmind.models.prediction import MotionPrediction, from_logits
from radarmind.paths import (
    CI4R_RESNET_PREFIX,
    CI4R_RESNET_RANDOM,
    MRI_GRU_RANDOM_XYZVI,
    MRI_GRU_SI_XYZ,
    RADHAR_GRU_XYZV,
    RADHAR_VOXEL,
    ROOT,
)
from radarmind.sensors.base import adapt_sample

_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Loaded:
    def __init__(self, spec: dict):
        self.spec = spec
        self.model = None
        self.device = _device()

    def ensure(self):
        if self.model is not None:
            return
        ckpt = Path(self.spec["checkpoint"])
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt}")
        kind = self.spec["kind"]
        if kind == "resnet18":
            from models.resnet18_spec import ResNet18Spec

            net = ResNet18Spec(self.spec["num_classes"], 1)
        elif kind == "pointnet_gru":
            from models.point_encoder import PointNetGRU

            net = PointNetGRU(self.spec["num_classes"], in_ch=self.spec["in_ch"])
        elif kind == "voxel_cnn_gru":
            from models.point_encoder import VoxelCNNGRU

            net = VoxelCNNGRU(self.spec["num_classes"])
        elif kind == "shared_motion":
            sys.path.insert(0, str(ROOT / "scripts"))
            sys.path.insert(0, str(ROOT / "scripts" / "cross_radar"))
            from train_cross_radar import EmbNet

            net = EmbNet(self.spec["num_classes"])
        else:
            raise ValueError(kind)
        try:
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=self.device)
        net.load_state_dict(state)
        net.to(self.device)
        net.eval()
        self.model = net

    @torch.no_grad()
    def predict_sample(self, sample: RadarSample) -> MotionPrediction:
        self.ensure()
        kind = self.spec["kind"]
        if kind == "resnet18":
            inp = adapt_sample(sample)
            x = torch.from_numpy(inp.tensor).float().unsqueeze(0).to(self.device)
            logits, emb = _resnet_forward(self.model, x)
        elif kind == "shared_motion":
            inp = adapt_sample(sample)
            x = torch.from_numpy(inp.tensor).float().unsqueeze(0).to(self.device)
            logits, emb = _embnet_forward(self.model, x)
        elif kind == "pointnet_gru":
            n_ch = int(self.spec["in_ch"])
            inp = adapt_sample(sample, n_ch=n_ch)
            x = torch.from_numpy(inp.tensor).float().unsqueeze(0).to(self.device)
            mask = torch.from_numpy(np.asarray(inp.mask, dtype=bool)).unsqueeze(0).to(self.device)
            logits, emb = _gru_forward(self.model, x, mask)
        else:
            raise RuntimeError("voxel predict goes through predict_voxel")
        pred = from_logits(
            logits,
            self.spec["classes"],
            self.spec["name"],
            sample.dataset,
            self.spec["protocol"],
            sample.sample_id,
            embedding=emb,
        )
        pack = self.spec.get("open_set")
        if pack:
            pred = apply_open_set(pred, pack)
        return pred


def _embnet_forward(net, x):
    logits, emb = net(x, return_emb=True)
    return logits[0].cpu().numpy(), emb[0].cpu().numpy()


def _resnet_forward(net, x):
    bb = net.backbone
    h = bb.conv1(x if x.ndim == 4 else x.unsqueeze(1))
    h = bb.bn1(h)
    h = bb.relu(h)
    h = bb.maxpool(h)
    h = bb.layer1(h)
    h = bb.layer2(h)
    h = bb.layer3(h)
    h = bb.layer4(h)
    h = bb.avgpool(h)
    emb = torch.flatten(h, 1)
    logits = bb.fc(emb)
    return logits[0].cpu().numpy(), emb[0].cpu().numpy()


def _gru_forward(net, x, mask):
    h = net.enc(x, mask)
    out, _ = net.gru(h)
    emb = out[:, -1]
    logits = net.fc(emb)
    return logits[0].cpu().numpy(), emb[0].cpu().numpy()


MODEL_SPECS = {
    "ci4r_resnet18": {
        "name": "ci4r_resnet18",
        "kind": "resnet18",
        "dataset": "ci4r",
        "protocol": "within_distribution",
        "checkpoint": str(CI4R_RESNET_RANDOM),
        "num_classes": 11,
        "classes": CI4R_CLASSES,
        "features": "micro_doppler_224",
        "in_ch": 1,
    },
    "ci4r_resnet18_prefix": {
        "name": "ci4r_resnet18_prefix",
        "kind": "resnet18",
        "dataset": "ci4r",
        "protocol": "prefix_group_disjoint",
        "checkpoint": str(CI4R_RESNET_PREFIX),
        "num_classes": 11,
        "classes": CI4R_CLASSES,
        "features": "micro_doppler_224",
        "in_ch": 1,
    },
    "mri_pointnet_gru": {
        "name": "mri_pointnet_gru",
        "kind": "pointnet_gru",
        "dataset": "mri",
        "protocol": "subject_independent",
        "checkpoint": str(MRI_GRU_SI_XYZ),
        "num_classes": 13,
        "classes": MRI_CLASSES,
        "features": "XYZ",
        "in_ch": 3,
    },
    "mri_pointnet_gru_random": {
        "name": "mri_pointnet_gru_random",
        "kind": "pointnet_gru",
        "dataset": "mri",
        "protocol": "within_distribution",
        "checkpoint": str(MRI_GRU_RANDOM_XYZVI),
        "num_classes": 13,
        "classes": MRI_CLASSES,
        "features": "XYZVI",
        "in_ch": 5,
    },
    "radhar_pointnet_gru": {
        "name": "radhar_pointnet_gru",
        "kind": "pointnet_gru",
        "dataset": "radhar",
        "protocol": "official_split",
        "checkpoint": str(RADHAR_GRU_XYZV),
        "num_classes": 5,
        "classes": RAD_CLASSES,
        "features": "XYZV",
        "in_ch": 4,
    },
    "shared_motion_encoder_v01": {
        "name": "shared_motion_encoder_v01",
        "kind": "shared_motion",
        "dataset": "ci4r77",
        "datasets": ["ci4r10", "ci4r24", "ci4r77"],
        "protocol": "multi_radar_shared",
        "checkpoint": str(ROOT / "results" / "phase3_shared_motion" / "runs" / "M2_shared" / "best.pt"),
        "num_classes": 7,
        "classes": ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"],
        "features": "official_png_224",
        "in_ch": 1,
    },
    "radhar_voxel_cnn_gru": {
        "name": "radhar_voxel_cnn_gru",
        "kind": "voxel_cnn_gru",
        "dataset": "radhar",
        "protocol": "official_split",
        "checkpoint": str(RADHAR_VOXEL),
        "num_classes": 5,
        "classes": RAD_CLASSES,
        "features": "occupancy",
        "in_ch": 10,
    },
}

def _register_open_set_specs() -> None:
    pack = load_app_pack(ROOT / "results" / "phase5_open_set" / "app_pack.json")
    if not pack:
        return
    ckpt = Path(pack["checkpoint"])
    if not ckpt.is_file():
        return
    MODEL_SPECS["mri_openset"] = {
        "name": "mri_openset",
        "kind": "pointnet_gru",
        "dataset": "mri",
        "protocol": "open_set_loao",
        "checkpoint": str(ckpt),
        "num_classes": len(pack["known_classes"]),
        "classes": pack["known_classes"],
        "features": "XYZ",
        "in_ch": 3,
        "open_set": pack,
    }
    PROTOCOL_MODEL[("mri", "open_set_loao")] = "mri_openset"


DEFAULT_MODEL = {
    "ci4r": "ci4r_resnet18",
    "mri": "mri_pointnet_gru",
    "radhar": "radhar_pointnet_gru",
}
BAND_DEFAULT = {
    "ci4r10": "shared_motion_encoder_v01",
    "ci4r24": "shared_motion_encoder_v01",
    "ci4r77": "shared_motion_encoder_v01",
}

PROTOCOL_MODEL = {
    ("ci4r", "within_distribution"): "ci4r_resnet18",
    ("ci4r", "random_record"): "ci4r_resnet18",
    ("ci4r", "prefix_group_disjoint"): "ci4r_resnet18_prefix",
    ("ci4r", "prefix_tail4_disjoint"): "ci4r_resnet18_prefix",
    ("mri", "subject_independent"): "mri_pointnet_gru",
    ("mri", "within_distribution"): "mri_pointnet_gru_random",
    ("mri", "random_segment"): "mri_pointnet_gru_random",
    ("radhar", "official_split"): "radhar_pointnet_gru",
    ("radhar", "official_train_test"): "radhar_pointnet_gru",
    ("ci4r10", "multi_radar_shared"): "shared_motion_encoder_v01",
    ("ci4r24", "multi_radar_shared"): "shared_motion_encoder_v01",
    ("ci4r77", "multi_radar_shared"): "shared_motion_encoder_v01",
}

_register_open_set_specs()


class ModelRegistry:
    def __init__(self):
        self._loaded: dict[str, _Loaded] = {}

    def names(self) -> list[str]:
        return [k for k, spec in MODEL_SPECS.items() if Path(spec["checkpoint"]).is_file()]

    def names_for_dataset(self, dataset: str) -> list[str]:
        out = []
        for k, spec in MODEL_SPECS.items():
            ds = spec.get("datasets") or [spec.get("dataset")]
            if dataset in ds and Path(spec["checkpoint"]).is_file():
                out.append(k)
        return out

    def missing(self) -> list[dict]:
        out = []
        for k, spec in MODEL_SPECS.items():
            if not Path(spec["checkpoint"]).is_file():
                out.append({"name": k, "checkpoint": spec["checkpoint"], "required": k in DEFAULT_MODEL.values()})
        return out

    def spec(self, name: str) -> dict:
        if name not in MODEL_SPECS:
            raise KeyError(name)
        return MODEL_SPECS[name]

    def resolve(self, dataset: str, protocol: str | None = None, model_name: str | None = None) -> str:
        if model_name:
            return model_name
        if protocol:
            hit = PROTOCOL_MODEL.get((dataset, protocol.strip().lower()))
            if hit:
                return hit
        if dataset in BAND_DEFAULT:
            return BAND_DEFAULT[dataset]
        return DEFAULT_MODEL[dataset]

    def _get(self, name: str) -> _Loaded:
        if name not in self._loaded:
            self._loaded[name] = _Loaded(MODEL_SPECS[name])
        return self._loaded[name]

    def predict(self, sample: RadarSample, model_name: str | None = None, protocol: str | None = None) -> MotionPrediction:
        name = self.resolve(sample.dataset, protocol=protocol, model_name=model_name)
        wrapper = self._get(name)
        if wrapper.spec["kind"] == "voxel_cnn_gru":
            raise RuntimeError("voxel model needs occupancy tensor; use predict_voxel")
        return wrapper.predict_sample(sample)

    def checkpoint_ok(self, name: str) -> bool:
        return Path(MODEL_SPECS[name]["checkpoint"]).is_file()
