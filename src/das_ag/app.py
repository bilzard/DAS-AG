import gc
import math
import os
import random
from dataclasses import dataclass

import clip
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as TF
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

from das_ag.aesthetic_predictor import AestheticPredictor


class AestheticModel(nn.Module):
    def __init__(self):
        super().__init__()

        assert AESTHETIC_PREDICTOR_PATH is not None
        assert CLIP_MODEL_PATH is not None

        self.model_path = AESTHETIC_PREDICTOR_PATH
        self.clip_model_path = CLIP_MODEL_PATH
        self.predictor, self.clip_model, self.preprocess = self.load_()
        self.pos_text_feature = None
        self.neg_text_feature = None

    def load_(self):
        state_dict = torch.load(self.model_path, weights_only=True, map_location="cpu")
        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        clip_model, preprocess = clip.load(self.clip_model_path, device="cpu")

        return predictor, clip_model, preprocess

    def tokenize(self, text: list[str]):
        return clip.tokenize(text)

    def encode_image(self, image, normalize=True):
        x = self.clip_model.encode_image(image)
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode_text(self, text, normalize=True):
        x = self.clip_model.encode_text(text)
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x

    def calc_scores(self, image, pos_texts: list[str], neg_texts: list[str]):
        image_features = self.encode_image(image)

        if len(pos_texts) > 0:
            if self.pos_text_feature is None:
                tokenized_pos_texts = clip.tokenize(pos_texts).to(image.device)
                pos_text_features = self.encode_text(tokenized_pos_texts)
                self.pos_text_feature = pos_text_features
            else:
                pos_text_features = self.pos_text_feature
            pos_clip_score = (image_features @ pos_text_features.T).mean()
        else:
            pos_clip_score = torch.zeros(1, device=image.device)

        if len(neg_texts) > 0:
            if self.neg_text_feature is None:
                tokenized_neg_texts = clip.tokenize(neg_texts).to(image.device)
                neg_text_features = self.encode_text(tokenized_neg_texts)
                self.neg_text_feature = neg_text_features
            else:
                neg_text_features = self.neg_text_feature

            neg_clip_score = (image_features @ neg_text_features.T).mean()
        else:
            neg_clip_score = torch.zeros(1, device=image.device)

        aesthetic_score = self.predictor(image_features).mean() / 10.0
        return aesthetic_score, pos_clip_score, neg_clip_score


def is_deterministic_algorithm_enabled():
    return (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":16:8"
    )


def seed_everything(seed: int = 42, use_deterministic_algorithm: bool = False):
    """Set the same seed for reproducibility across random, numpy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithm)


def interpolate(image, target_size, mode="bicubic"):
    return TF.interpolate(
        image,
        size=(target_size, target_size),
        mode=mode,
        **({"align_corners": False} if mode == "bicubic" or mode == "bilinear" else {}),
    )


def add_positional_rolling(image, max_shift=10):
    """
    `torch.roll()` を使って勾配を維持したまま Positional Jitter を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :param max_shift: 最大移動ピクセル数
    :return: ジッターを加えた画像
    """
    N, C, H, W = image.shape
    jittered_images = torch.zeros_like(image)

    # 各画像にランダムなシフトを適用
    for i in range(N):
        dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()  # X 方向のシフト
        dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()  # Y 方向のシフト

        # **torch.roll() を使って画像をシフト（勾配を維持）**
        jittered_images[i] = torch.roll(image[i], shifts=(dy, dx), dims=(1, 2))  # type: ignore

    return jittered_images


def apply_cutout(image, max_size=50):
    """
    `cutout` を適用してランダムな領域をマスク
    :param image: (N, C, H, W) の PyTorch Tensor
    :param max_size: 切り取る最大サイズ
    :return: `cutout` を適用した画像
    """
    N, C, H, W = image.shape
    cutout_image = image.clone()

    for i in range(N):
        # **ランダムな位置を選択**
        x = torch.randint(0, W - max_size, (1,)).item()
        y = torch.randint(0, H - max_size, (1,)).item()
        w = torch.randint(10, max_size, (1,)).item()
        h = torch.randint(10, max_size, (1,)).item()

        # **選択した領域をゼロ（黒）にする**
        cutout_image[i, :, y : y + h, x : x + w] = 0

    return cutout_image


def color_shift(image, shift: float = 1.0):
    """
    画像に Color Shift を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :return: Color Shift された画像
    """
    N, C, H, W = image.shape
    mu = torch.zeros((N, C, 1, 1), device=image.device).uniform_(
        -shift, shift
    )  # U[-1,1]
    sigma = torch.exp(
        torch.zeros((N, C, 1, 1), device=image.device).uniform_(-shift, shift)
    )  # exp(U[-1,1])

    return sigma * image + mu  # カラースケール & シフト


def gaussian_noise(image, sigma=1.0):
    """
    画像に Gaussian Smoothing を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :return: ノイズを加えた画像
    """
    noise = torch.randn_like(image)  # N(0,1) のノイズを生成
    return image + sigma * noise  # 画像にノイズを加える


def gaussian_blur(image, kernel_size=(3, 3), sigma=(0.1, 2.0)):
    return F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def random_posterize(image, bits=3):
    return F2.posterize(image, bits)


def total_variation_loss(image):
    """
    Total Variation (TV) Loss を計算
    :param image: (1, 3, H, W) の PyTorch Tensor
    :return: TV Loss のスカラー値
    """
    dx = torch.diff(image, dim=2).abs().mean()  # 横方向の変化
    dy = torch.diff(image, dim=3).abs().mean()  # 縦方向の変化
    return dx + dy


def l1_regularization(image):
    """
    L1 正則化 (スパース性を強調)
    :param image: (1, 3, H, W) の PyTorch Tensor
    :return: L1 Loss のスカラー値
    """
    return image.abs().mean()


def linear_schedule(step, max_steps, start=0.5, end=0.0, **kwargs):
    """
    線形スケジューリング (0.5 → 0)
    :param step: 現在のステップ
    :param max_steps: 総ステップ数
    :param start: 初期ノイズ強度
    :param end: 最終ノイズ強度
    :return: スケジュールされたノイズの標準偏差
    """
    return start + (end - start) * (step / max_steps)


def exponential_decay_schedule(step, max_steps, start=0.5, end=0.0, rate=0.1, **kwargs):
    return (start - end) * (1 - rate) ** step + end


def exponential_warmup_schedule(
    step, max_steps, start=0.5, end=0.0, rate=0.03, **kwargs
):
    initial_bias = (end - start) * (1 - rate) ** max_steps
    return (end - start) * (1 - rate) ** (max_steps - step) + start - initial_bias


def cosine_schedule(step, max_steps, start=0.5, end=0.0, **kwargs):
    """
    コサイン減衰スケジューリング (0.5 → 0)

    :param step: 現在のステップ
    :param max_steps: 総ステップ数
    :param start: 初期ノイズ強度
    :param end: 最終ノイズ強度
    :return: スケジュールされたノイズの標準偏差
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return end + (start - end) * cosine_decay


def inv_process(image):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    inv_process = T.Compose(
        [
            T.Normalize(
                mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            ),
            T.ToPILImage(),
        ]
    )
    return inv_process(image)


def plot_histogram(image):
    import matplotlib.pyplot as plt

    plt.hist(image[..., 0].flatten(), color="r", label="R", alpha=0.3)
    plt.hist(image[..., 1].flatten(), color="g", label="G", alpha=0.3)
    plt.hist(image[..., 2].flatten(), color="b", label="B", alpha=0.3)
    plt.legend()
    plt.show()


@dataclass
class Config:
    num_steps: int = 100
    batch_size: int = 8
    lambda_tv: float = 1e-2
    lambda_l1: float = 0.05
    lr: float = 0.1
    betas: tuple[float, float] = (0.5, 0.99)
    eta_min_ratio: float = 0.01
    eval_steps: int = 1
    max_shift: int = 32
    noise_schedule: str = "exponential_decay"
    noise_decay_rate: float = 0.03
    color_shift_schedule: str = "exponential_decay"
    color_shift_decay_rate: float = 0.03
    noise_std_range: tuple[float, float] = (0.05, 0.5)
    color_shift_range: tuple[float, float] = (0.1, 1.0)
    image_resolutions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    checkpoint_interval: int = 100
    mode: str = "bilinear"
    seed: int = 42
    use_deterministic_algorithm: bool = False
    full_size: int = 224
    apply_augmentation: bool = True

    lambda_clip: float = 0.5
    aesthetic_range: tuple[float, float] = (0.0, 1.0)
    aesthetic_schedule: str = "exponential_warmup"
    aesthetic_decay_rate: float = 0.03
    reverse_aesthetic: bool = False


schedule_map = {
    "linear": linear_schedule,
    "exponential_warmup": exponential_warmup_schedule,
    "exponential_decay": exponential_decay_schedule,
    "cosine": cosine_schedule,
}


class DasAttacker:
    def __init__(
        self,
        config: Config,
        device="cuda",
    ):
        self.model = AestheticModel().to(device)

        self.device = device
        self.images = None
        self.config = config
        self.mode = config.mode
        self.checkpoints = []
        self.scores = []
        self.noise_schedule = schedule_map[config.noise_schedule]
        self.color_shift_schedule = schedule_map[config.color_shift_schedule]
        self.aesthetic_schedule = schedule_map[config.aesthetic_schedule]
        self.max_size = self.config.image_resolutions[-1]
        self.full_size = self.config.full_size
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def attack(
        self, pos_texts: list[str], neg_texts: list[str], progress_callback=None
    ):
        cfg = self.config
        image_stack = [
            (nn.Parameter(torch.randn(1, 3, s, s, device=self.device) / s))
            for s in cfg.image_resolutions
        ]
        optimizer = torch.optim.Adam(image_stack, lr=cfg.lr, betas=cfg.betas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_steps, eta_min=cfg.eta_min_ratio * cfg.lr
        )
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            noise_std = self.noise_schedule(
                step,
                cfg.num_steps,
                start=cfg.noise_std_range[0],
                end=cfg.noise_std_range[1],
                rate=cfg.noise_decay_rate,
            )
            c_shift = self.color_shift_schedule(
                step,
                cfg.num_steps,
                start=cfg.color_shift_range[0],
                end=cfg.color_shift_range[1],
                rate=cfg.color_shift_decay_rate,
            )
            lambda_aesthetic = self.aesthetic_schedule(
                step,
                cfg.num_steps,
                start=cfg.aesthetic_range[0],
                end=cfg.aesthetic_range[1],
                rate=cfg.aesthetic_decay_rate,
            )
            image = torch.stack(
                [interpolate(i, self.max_size, mode=self.mode) for i in image_stack]
            ).mean(0)
            image = image.tanh()
            if cfg.apply_augmentation:
                images = image.repeat(cfg.batch_size, 1, 1, 1)

                images = add_positional_rolling(images, max_shift=cfg.max_shift)
                images = color_shift(images, c_shift)
                images = gaussian_noise(images, sigma=noise_std)
            else:
                images = image.repeat(1, 1, 1, 1)

            self.pad = pad = cfg.max_shift
            images = images[..., pad:-pad, pad:-pad]
            if images.shape[-1] != self.full_size:
                images = interpolate(images, self.full_size, mode=self.mode)

            aesthetic_score, pos_clip, neg_clip = self.model.calc_scores(
                images, pos_texts, neg_texts
            )

            if cfg.reverse_aesthetic:
                lambda_aesthetic = -lambda_aesthetic

            loss_tv = total_variation_loss(images)
            loss_l1 = l1_regularization(images)
            loss = (
                -lambda_aesthetic * aesthetic_score
                - cfg.lambda_clip * (pos_clip - neg_clip)
                + cfg.lambda_tv * loss_tv
                + cfg.lambda_l1 * loss_l1
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            if progress_callback:
                progress_callback((step + 1) / cfg.num_steps)

            if (step + 1) % cfg.checkpoint_interval == 0:
                images_list = []
                image_tensor = torch.cat(
                    [
                        interpolate(i.detach(), self.max_size, mode=self.mode).cpu()
                        for i in image_stack
                    ]
                )
                for i in range(len(image_tensor)):
                    images_list.append(image_tensor[i])
                self.checkpoints.append(torch.stack(images_list))
                self.scores.append(aesthetic_score.mean().item())

        self.image_stack = [i.detach().cpu() for i in image_stack]
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self, pos_texts, neg_texts):
        with torch.no_grad():
            image = torch.cat(
                [
                    interpolate(i, self.max_size, mode=self.mode).to(self.device)
                    for i in self.image_stack
                ]
            ).mean(0)
            image = image[..., self.pad : -self.pad, self.pad : -self.pad]
            image = image.tanh()
            image_np = inv_process(image.squeeze(0).detach().cpu())

            image = image.unsqueeze(0)
            if image.shape[-1] != self.full_size:
                image = interpolate(image, self.full_size, mode=self.mode)

            aesthetic_score, pos_clip_score, neg_clip_score = self.model.calc_scores(
                image, pos_texts, neg_texts
            )
            return (
                image_np,
                aesthetic_score,
                pos_clip_score,
                neg_clip_score,
            )


def main():
    st.title("Aesthetic Image Generation with CLIP")

    # Positive Prompts
    st.header("Positive Prompts")
    if "positive_prompts" not in st.session_state:
        st.session_state.positive_prompts = [
            'A photorealistic illustration of "a stunning Mount Fuji, a majestic hawk, and a symbolic eggplant, set against the neon-lit skyline of cyberpunk Tokyo", dynamic camera angle, fine-grained details, well-recognizable, close-up'
        ]

    def add_pos_input():
        st.session_state.positive_prompts.append("")

    def remove_pos_input():
        if len(st.session_state.positive_prompts) > 0:
            st.session_state.positive_prompts.pop()

    for i in range(len(st.session_state.positive_prompts)):
        st.session_state.positive_prompts[i] = st.text_area(
            f"Prompt {i + 1}", st.session_state.positive_prompts[i]
        )
    col1, col2 = st.columns(2)
    with col1:
        st.button("Add", on_click=add_pos_input, key="add_pos")
    with col2:
        st.button("Remove", on_click=remove_pos_input, key="remove_pos")

    # Negative Prompts
    st.header("Negative Prompts")
    if "negative_prompts" not in st.session_state:
        st.session_state.negative_prompts = [
            "text present, low-quality, low-resolution, insane, ugly, grotesque, horrifying, blurred, noisy, distorted, pixelated, artifact present, without depth, flat, boring, uninteresting, unattractive, unappealing, unclear, dull, dark, gloomy, depressing, monotonous"
        ]

    def add_neg_input():
        st.session_state.negative_prompts.append("")

    def remove_neg_input():
        if len(st.session_state.negative_prompts) > 0:
            st.session_state.negative_prompts.pop()

    for i in range(len(st.session_state.negative_prompts)):
        st.session_state.negative_prompts[i] = st.text_area(
            f"Prompt {i + 1}", st.session_state.negative_prompts[i]
        )
    col1, col2 = st.columns(2)
    with col1:
        st.button("Add", on_click=add_neg_input, key="add_neg")
    with col2:
        st.button("Remove", on_click=remove_neg_input, key="remove_neg")

    # Sidebar
    st.sidebar.header("Parameters")
    resolution = st.sidebar.selectbox(
        "Resolution", [224, 448], index=0, help="Resolution of the generated image"
    )
    num_steps = st.sidebar.slider(
        "#steps", min_value=0, max_value=200, value=100, step=10
    )
    batch_size = st.sidebar.slider(
        "batch size", min_value=4, max_value=32, value=8, step=4
    )
    lr = st.sidebar.slider("lr", min_value=0.00, max_value=0.20, value=0.10, step=0.01)

    clip_weight = st.sidebar.slider(
        "CLIP Weight",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=1.0,
        help="Weight of CLIP loss",
    )
    st.sidebar.header("Aesthetic")
    reverse_aesthetic = st.sidebar.checkbox(
        "Reverse Aesthetic",
        value=False,
        help="If enabled, the aesthetic score will be reversed",
    )
    aesthetic_range = st.sidebar.slider(
        "Aesthetic Range",
        min_value=0.0,
        max_value=5.0,
        value=(0.0, 3.0),
        step=0.5,
    )
    aesthetic_schedule = st.sidebar.selectbox(
        "Aesthetic Schedule",
        list(schedule_map.keys()),
        index=2,
    )
    aesthetic_decay_rate = st.sidebar.slider(
        "Aesthetic Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.06,
        step=0.01,
        disabled=aesthetic_schedule == "linear",
    )

    st.sidebar.header("Generation")
    use_deterministic_algorithm = (
        st.sidebar.checkbox(
            "Use Deterministic Algorithm",
            value=False,
            help="**Note**: it does not ensure reproducibility and it becomes slower",
        )
        if is_deterministic_algorithm_enabled()
        else False
    )
    interpolation_mode = st.sidebar.selectbox(
        "Interpolation Mode",
        ["bilinear", "bicubic", "nearest"]
        if not use_deterministic_algorithm
        else ["bilinear"],
    )
    set_seed = st.sidebar.checkbox("Set Seed", value=False)
    seed = st.sidebar.number_input("seed", value=42, step=1, disabled=not set_seed)

    st.sidebar.header("Regularization")
    lambda_tv_exp = st.sidebar.slider(
        "Texture Suppression (TV)",
        min_value=-8,
        max_value=0,
        value=-8,
        help="higher value suppresses texture and increases smoothness",
    )
    lambda_tv = 10**lambda_tv_exp
    lambda_l1 = st.sidebar.slider(
        "Color Suppression (L1)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.05,
        help="higher value suppresses color and increases grayness",
    )

    st.sidebar.header("Augmentation")
    apply_augmentation = st.sidebar.checkbox("Apply Augmentation", value=True)
    st.sidebar.subheader("Gaussian Noise")
    noise_schedule = st.sidebar.selectbox(
        "Noise Schedule",
        list(schedule_map.keys()),
        index=1,
        disabled=not apply_augmentation,
    )
    noise_stds = st.sidebar.slider(
        "Noise Intensity Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.2, 0.5),
        step=0.05,
        disabled=not apply_augmentation,
    )
    noise_decay_rate = st.sidebar.slider(
        "Noise Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        disabled=not apply_augmentation or noise_schedule == "linear",
    )
    st.sidebar.subheader("Positional Jitter")
    st.sidebar.subheader("Color Shift")
    color_shift_schedule = st.sidebar.selectbox(
        "Color Shift Schedule",
        list(schedule_map.keys()),
        index=1,
        disabled=not apply_augmentation,
    )
    color_shift_range = st.sidebar.slider(
        "Color Shift Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.05, 0.30),
        step=0.05,
        disabled=not apply_augmentation,
    )
    color_shift_decay_rate = st.sidebar.slider(
        "Color Shift Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        disabled=not apply_augmentation or color_shift_schedule == "linear",
    )
    if resolution == 224:
        image_resolutions = (1, 2, 4, 8, 16, 32, 64, 128, 256)
        max_shift = 16
    elif resolution == 448:
        image_resolutions = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        max_shift = 32
    else:
        raise ValueError("Invalid resolution")

    cfg = Config(
        num_steps=num_steps,
        batch_size=batch_size,
        lambda_tv=lambda_tv,
        lambda_l1=lambda_l1,
        lr=lr,
        checkpoint_interval=num_steps,
        noise_schedule=noise_schedule,
        noise_std_range=noise_stds,
        noise_decay_rate=noise_decay_rate,
        color_shift_schedule=color_shift_schedule,
        color_shift_decay_rate=color_shift_decay_rate,
        color_shift_range=color_shift_range,
        seed=seed,
        mode=interpolation_mode,
        use_deterministic_algorithm=use_deterministic_algorithm,
        apply_augmentation=apply_augmentation,
        lambda_clip=clip_weight,
        aesthetic_range=aesthetic_range,
        aesthetic_schedule=aesthetic_schedule,
        aesthetic_decay_rate=aesthetic_decay_rate,
        image_resolutions=image_resolutions,
        max_shift=max_shift,
        reverse_aesthetic=reverse_aesthetic,
    )
    positive_prompts = st.session_state.positive_prompts
    negative_prompts = st.session_state.negative_prompts

    if st.button("Generate Image"):
        if len(positive_prompts) > 0:
            st.write("**Positive Prompts**:")
            for pos_prompt in positive_prompts:
                st.write(f"`{pos_prompt}`")

        if len(negative_prompts) > 0:
            st.write("**Negative Prompts**:")
            for neg_prompt in negative_prompts:
                st.write(f"`{neg_prompt}`")
            st.write("Generating Image...")

        if set_seed:
            seed_everything(
                cfg.seed,
                use_deterministic_algorithm=cfg.use_deterministic_algorithm,
            )
        attacker = DasAttacker(config=cfg)
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Progress: {int(progress * 100)}%")

        attacker.attack(
            positive_prompts, negative_prompts, progress_callback=update_progress
        )
        st.success("Image Generation Completed!")

        st.subheader("Result")
        result_img, aesthetic_score, pos_clip_score, neg_clip_score = attacker.evaluate(
            positive_prompts, negative_prompts
        )
        st.image(
            result_img,
            caption=f"Aesthetic Score: {aesthetic_score * 10:.2f}, CLIP Score: {pos_clip_score:.3f} (Negative: {neg_clip_score:.3f})",
            use_container_width=True,
            output_format="PNG",
        )


if __name__ == "__main__":
    AESTHETIC_PREDICTOR_PATH = os.environ.get("AESTHETIC_PREDICTOR_PATH")
    CLIP_MODEL_PATH = os.environ.get("CLIP_MODEL_PATH")
    main()
