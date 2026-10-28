# ==============================================================
# SmolVLA + Flow Matching（reference/basic_final_assignment.ipynb の 8章を切り出したもの）
#   学習（notebooks/train.ipynb, Colab T4）と推論（crane_pp/infer.py, コンテナ CPU）の両方で import する。
#   ノートブックからの変更点:
#     - VLM のロードを load_vlm_encoder() 関数にした（GPU/CPU と dtype を呼び出し側で選ぶため）
#     - TODO（問題箇所）は解答済みのコードにし、動作確認用のコードは省いた
#     - action_dim / state_dim の既定値を CRANE+ V2 の 5 次元にした
#     - batch>1 で学習できるよう、processor に画像をサンプルごとのリストで渡すようにした
#     - GPU メモリと計算量を減らすため、画像のタイル分割をやめた（do_image_splitting=False）
# ==============================================================
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# SmolVLA の Vision-Language backbone（500M パラメータ版）
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


# ==============================================================
# 8.1 時刻埋め込み (SinusoidalTimeEmbedding)
#   連続時刻 t ∈ [0, 1] を hidden_dim 次元の sin/cos ベクトルにする
#     embed(t)[2i]   = sin(t / 10000^(2i/D))
#     embed(t)[2i+1] = cos(t / 10000^(2i/D))
# ==============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """スカラー時刻 t を hidden_dim 次元のベクトルに埋め込む."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # sin と cos で半分ずつ埋めるので偶数を要求
        assert hidden_dim % 2 == 0, "hidden_dim は偶数にしてください"
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 入力 t: (B,)  →  出力: (B, hidden_dim)
        half = self.hidden_dim // 2
        # 周波数の等比数列: 1 / 10000^(i/half)。高周波 → 低周波
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        # 時刻 × 周波数 の外積 (B, half)
        args = t[:, None] * freqs[None, :]
        # sin と cos を横に連結 (B, hidden_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ==============================================================
# 8.2 Transformer ブロック（Pre-LN 型）
#   x -> LN -> self_attn -> +残差 -> LN -> cross_attn(VLM 参照) -> +残差 -> LN -> MLP -> +残差
#   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
# ==============================================================
class TransformerBlock(nn.Module):
    """Transformer decoder 1 層."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        # multi-head の分割で余りが出ないよう割り切れる必要あり
        assert hidden_dim % num_heads == 0, "hidden_dim は num_heads の倍数にしてください"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention の Q/K/V/O（action tokens 同士の相互参照）
        self.self_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cross-attention の Q/K/V/O（Q は action tokens、K/V は VLM 出力 = 画像+言語+state の条件）
        self.cross_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP（D → 4D → GELU → D）
        self.mlp_in = nn.Linear(hidden_dim, mlp_ratio * hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_ratio * hidden_dim, hidden_dim, bias=False)

        # Pre-LN 用の LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) → (B, num_heads, L, head_dim)
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, L, head_dim) → (B, L, D)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _attention(self, q_proj, k_proj, v_proj, o_proj, q_input, k_input, v_input):
        # 射影 → head 分割 → Scaled Dot-Product Attention → head 統合 → 出力射影
        Q = self._split_heads(q_proj(q_input))
        K = self._split_heads(k_proj(k_input))
        V = self._split_heads(v_proj(v_input))
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        return o_proj(self._merge_heads(attn_out))

    def forward(self, action_tokens: torch.Tensor, vlm_hidden: torch.Tensor) -> torch.Tensor:
        # action_tokens: (B, chunk, D)、vlm_hidden: (B, T_vlm, D)  →  (B, chunk, D)
        # Sub-layer 1: Self-attention（chunk 内の相互作用）
        h = self.norm1(action_tokens)
        action_tokens = action_tokens + self._attention(
            self.self_q_proj, self.self_k_proj, self.self_v_proj, self.self_o_proj, h, h, h)
        # Sub-layer 2: Cross-attention（VLM 参照 = 条件付け）
        h = self.norm2(action_tokens)
        action_tokens = action_tokens + self._attention(
            self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.cross_o_proj,
            h, vlm_hidden, vlm_hidden)
        # Sub-layer 3: Position-wise MLP
        h = self.norm3(action_tokens)
        action_tokens = action_tokens + self.mlp_out(F.gelu(self.mlp_in(h)))
        return action_tokens


# ==============================================================
# 8.3 Flow Matching Head
#   noise → action への速度場 vθ(x(t), t, cond) を予測する Transformer
#   (B, K, d_a) → action_in → +pos_emb → +time_emb → Transformer × N → action_out → (B, K, d_a)
# ==============================================================
class FlowMatchingHead(nn.Module):
    """action chunk の velocity 場を条件付きで予測する Transformer."""

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int, num_layers: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # action 次元 ⇄ hidden 次元 の入出力射影
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        # chunk 内の位置埋め込み（learnable, zero-init）
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        # 時刻 t 用の埋め込み器
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        # Transformer 本体
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, vlm_hidden: torch.Tensor) -> torch.Tensor:
        # noisy_actions: (B, K, d_a)、t: (B,)、vlm_hidden: (B, T_vlm, D)  →  velocity (B, K, d_a)
        x = self.action_in(noisy_actions) + self.pos_embed  # (B, K, D)
        x = x + self.time_embed(t)[:, None, :]  # 時刻埋め込みを chunk 全 step に加算
        for block in self.blocks:
            x = block(x, vlm_hidden)
        return self.action_out(x)


# ==============================================================
# 8.4 SmolVLM2 をロードして完全に凍結する
# ==============================================================
class RealVLMEncoder(nn.Module):
    """SmolVLM2 をラップ: (画像 list, task 文 list) → 最終層の hidden state (B, L, hidden_dim)."""

    def __init__(self, model, processor, hidden_dim: int, dtype):
        super().__init__()
        self.model = model
        self.processor = processor
        self.hidden_dim = hidden_dim
        self._dtype = dtype

    @torch.no_grad()
    def forward(self, images, task_texts: list[str]) -> torch.Tensor:
        # 入力を PIL に統一（processor は PIL を期待する）
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            elif isinstance(img, torch.Tensor):
                arr = img.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] == 3:  # (C,H,W) → (H,W,C)
                    arr = arr.transpose(1, 2, 0)
                pil_images.append(Image.fromarray(arr.astype(np.uint8)))
            else:
                pil_images.append(img)

        # SmolVLM2 の chat template で「画像 + 指示文」の prompt に組み立てる
        prompts = []
        for txt in task_texts:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}]
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))

        # tokenize + 画像前処理。padding=True で可変長を揃える
        # images は「サンプルごとの画像リスト」で渡す（[[img0], [img1], ...]）。
        # リファレンスは batch=1 だったので平らなリストでも動いたが、batch>1 だと
        # 「テキスト中の画像数 [1, 1, ...] と画像数 [B] が合わない」エラーになるため
        # do_image_splitting=False: 画像をタイルに分割せず 1 枚として扱う（画像トークン 64 個）。
        # 既定の分割ありだと 256x256 でも拡大・分割されてトークンが十数倍になり、
        # Colab T4 で batch=8 だと GPU メモリ不足（OOM）になったため。CPU 推論も速くなる
        inputs = self.processor(
            text=prompts, images=[[img] for img in pil_images], return_tensors="pt", padding=True,
            do_image_splitting=False)

        # モデルの device / dtype に揃える（float だけ dtype 変換し、token id などはそのまま）
        device = next(self.model.parameters()).device
        inputs = {
            k: (v.to(device=device, dtype=self._dtype)
                if v.dtype in (torch.float16, torch.float32, torch.bfloat16)
                else v.to(device))
            for k, v in inputs.items()
        }

        # 最終 layer の hidden state（画像 tokens + テキスト tokens の融合表現）を返す
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]


def load_vlm_encoder(device: str, dtype) -> RealVLMEncoder:
    """SmolVLM2 を読み込み、完全に凍結して RealVLMEncoder に包む.

    学習（Colab T4）は device="cuda", dtype=torch.float16、推論（CPU）は device="cpu", dtype=torch.float32 を想定。
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # processor: 画像リサイズ・正規化・トークン化を担当
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    # transformers 5 系では torch_dtype ではなく dtype で指定する
    model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL_ID, dtype=dtype).to(device)
    # 完全 freeze（VLM は学習対象外）
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.text_config.hidden_size  # 960
    return RealVLMEncoder(model, processor, hidden_dim, dtype)


# ==============================================================
# 8.5 SmolVLA 統合モジュール
#   学習 (compute_loss):
#     L(θ) = || vθ(x(t), t, cond) − (a − x0) ||²
#     x0 ~ N(0, I),  t ~ U(0, 1),  x(t) = (1 − t) × x0 + t × a
#     cond = [vlm_projector(VLM(images, task_text)), state_proj(state)]
#   推論 (sample_actions, Euler 積分 t: 0 → 1):
#     x ← N(0, I) からサンプル
#     for step in 0..N−1:  x ← x + vθ(x, step/N, cond) × (1/N)
# ==============================================================
class SmolVLA(nn.Module):
    """SmolVLA 全体. VLM は frozen、Action head は trainable、state は projection 経由."""

    def __init__(
        self,
        vlm_encoder: RealVLMEncoder,
        action_dim: int = 5,  # CRANE+ V2: joint1〜4 + gripper
        state_dim: int = 5,  # 同上（/joint_states）
        chunk_size: int = 10,  # 予測する連続 action 数（10 step = 1 秒 @10Hz）
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        transformer_hidden_dim: int = 256,  # VLM の 960 から 256 に射影して軽くする
    ):
        super().__init__()
        self.vlm_encoder = vlm_encoder
        # VLM 出力 (960) を Transformer 用の次元に線形射影
        self.vlm_projector = nn.Linear(vlm_encoder.hidden_dim, transformer_hidden_dim, bias=False)
        # state を 1 トークンに射影する層（VLA の "S" 部分）
        self.state_proj = nn.Linear(state_dim, transformer_hidden_dim)

        self.hidden_dim = transformer_hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size

        self.flow_head = FlowMatchingHead(
            hidden_dim=transformer_hidden_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
        )

    def _encode_cond(self, images, task_texts, states: torch.Tensor) -> torch.Tensor:
        # 1) VLM から画像+言語の融合表現（勾配なし）
        hidden = self.vlm_encoder(images, task_texts)  # (B, L, 960)
        # 2) dtype を action head に合わせる（VLM: fp16, head: fp32）
        hidden = hidden.to(dtype=next(self.flow_head.parameters()).dtype)
        # 3) Transformer の hidden 次元に射影
        cond_vlm = self.vlm_projector(hidden)  # (B, L, D)
        # 4) state を射影して 1 トークンとして末尾に連結
        state_token = self.state_proj(states.to(dtype=cond_vlm.dtype, device=cond_vlm.device)).unsqueeze(1)
        return torch.cat([cond_vlm, state_token], dim=1)  # (B, L+1, D)

    def compute_loss(self, images, task_texts: list[str], states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Flow Matching 学習損失（actions は正規化済み (B, K, d_a)）."""
        B = actions.size(0)
        device = actions.device
        # 条件（VLM + state）
        vlm_hidden = self._encode_cond(images, task_texts, states).to(device)
        # ノイズ x0 ~ N(0, I) と時刻 t ~ U(0, 1)
        noise = torch.randn_like(actions)
        t = torch.rand(B, device=device)
        tv = t[:, None, None]
        # 線形補間で中間点 x(t) = (1 − t) × x0 + t × a
        xt = (1 - tv) * noise + tv * actions
        # 真の速度 v* = a − x0（直線パスなので一定）
        target_v = actions - noise
        # ネットワークで速度を予測して MSE
        pred_v = self.flow_head(xt, t, vlm_hidden)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample_actions(self, images, task_texts: list[str], states: torch.Tensor, num_steps: int = 10,
                       noise_scale: float = 1.0) -> torch.Tensor:
        """Euler 法で ODE を積分して action chunk (B, K, d_a)（正規化空間）を復元."""
        device = next(self.flow_head.parameters()).device
        # 条件は 1 回だけ計算して使い回す
        vlm_hidden = self._encode_cond(images, task_texts, states).to(device)
        B = vlm_hidden.size(0)
        # pure noise から開始
        # noise_scale=0 にすると初期値が 0 に固定され、同じ観測からは毎回同じ軌道が出る（推論のばらつきを抑える）
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device) * noise_scale
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_now = torch.full((B,), step * dt, device=device)
            # Euler 1 歩: x(t + Δt) = x(t) + vθ(x(t), t, cond) × Δt
            x = x + self.flow_head(x, t_now, vlm_hidden) * dt
        # 呼び出し側で std を掛けて mean を足し、実スケールに戻す
        return x
