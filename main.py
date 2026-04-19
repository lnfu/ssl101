from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image

from input_pipeline import _load_cifar10_split
from models import ResNet18

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NUM_CLASSES = 10
DISPLAY_SIZE = 320

_model: ResNet18 | None = None


def _discover_checkpoints(root: str = "checkpoints") -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    paths = sorted(
        p
        for p in base.glob("*/epoch_*")
        if p.name.startswith("epoch_") and p.name[6:].isdigit()
    )
    return [str(p) for p in paths]


def load_model(ckpt_path: str) -> str:
    global _model
    if not ckpt_path:
        return "⚠️ Select a checkpoint first."
    p = Path(ckpt_path).resolve()
    if not p.exists():
        return f"⚠️ Path not found: {p}"
    epoch = int(p.name.split("_")[1])
    model = ResNet18(num_classes=NUM_CLASSES, rngs=nnx.Rngs(42))
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    restored = checkpointer.restore(str(p), {"model_state": state, "epoch": 0})
    nnx.update(model, restored["model_state"])
    _model = model
    return f"✅ Loaded epoch {epoch}  ({p.parent.name})"


def _to_display(arr: np.ndarray) -> np.ndarray:
    """(32, 32, 3) float32 [0,1]  →  (DISPLAY_SIZE, DISPLAY_SIZE, 3) uint8"""
    u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return np.array(
        Image.fromarray(u8).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST
        )
    )


def _dataset_item(split: str, idx: int) -> tuple[np.ndarray, int, str]:
    X, y = _load_cifar10_split(split)
    idx = int(idx) % len(X)
    img = _to_display(X[idx])
    label = CIFAR10_CLASSES[y[idx]]
    info = f"{split}  |  index {idx} / {len(X) - 1}  |  true label: **{label}**"
    return img, idx, info


def classify(img: np.ndarray | None) -> str:
    if _model is None:
        return "⚠️ No model loaded. Please load a checkpoint first."
    if img is None:
        return "⚠️ No image available."

    pil = Image.fromarray(img.astype(np.uint8)).resize(
        (32, 32), Image.Resampling.BILINEAR
    )
    x = jnp.array(np.array(pil, dtype=np.float32) / 255.0)[None]  # (1,32,32,3)

    logits = _model(x, use_running_average=True)
    probs = jax.nn.softmax(logits[0])
    pred = int(jnp.argmax(probs))

    confidence = f"{float(probs[pred]):.1%}"
    lines = [f"## 🏷️  {CIFAR10_CLASSES[pred].upper()}  —  {confidence} confidence\n"]
    for i, (cls, p) in enumerate(zip(CIFAR10_CLASSES, probs)):
        pct = float(p)
        filled = int(pct * 24)
        bar = "█" * filled + "░" * (24 - filled)
        arrow = "  ◀" if i == pred else ""
        lines.append(f"`{cls:<12}` `{bar}` {pct:5.1%}{arrow}")
    return "  \n".join(lines)


def build_app() -> gr.Blocks:
    checkpoints = _discover_checkpoints()

    with gr.Blocks(title="CIFAR-10 ResNet18") as demo:
        gr.Markdown("# CIFAR-10 ResNet18 Demo")

        # checkpoint loader
        with gr.Row():
            ckpt_dd = gr.Dropdown(
                choices=checkpoints,
                value=checkpoints[-1] if checkpoints else None,
                label="Checkpoint",
                scale=5,
            )
            load_btn = gr.Button("Load Model", scale=1, variant="secondary")
            load_status = gr.Textbox(label="Status", interactive=False, scale=4)

        load_btn.click(fn=load_model, inputs=[ckpt_dd], outputs=[load_status])

        gr.Markdown("---")

        # main panel
        with gr.Row():
            with gr.Column(scale=1, min_width=DISPLAY_SIZE + 40):
                display_img = gr.Image(
                    label="Current Image",
                    type="numpy",
                    interactive=False,
                    height=DISPLAY_SIZE,
                    width=DISPLAY_SIZE,
                )

                with gr.Tabs():
                    # Dataset tab
                    with gr.Tab("📂  Dataset"):
                        split_radio = gr.Radio(
                            choices=["train", "test"],
                            value="test",
                            label="Split",
                        )
                        idx_state = gr.State(value=0)
                        idx_info = gr.Markdown("*(loading…)*")
                        with gr.Row():
                            prev_btn = gr.Button("◀  Previous", size="sm")
                            next_btn = gr.Button("Next  ▶", size="sm")

                    # Upload tab
                    with gr.Tab("📤  Upload"):
                        upload_img = gr.Image(
                            label="Drag & drop or click to upload",
                            type="numpy",
                            sources=["upload"],
                            height=200,
                        )

            with gr.Column(scale=1):
                classify_btn = gr.Button(
                    "🔍   CLASSIFY",
                    variant="primary",
                    size="lg",
                    elem_id="classify-btn",
                )
                result_md = gr.Markdown("*(press Classify to see the prediction)*")

        def on_split_change(split):
            img, idx, info = _dataset_item(split, 0)
            return img, idx, info

        def on_prev(split, idx):
            img, new_idx, info = _dataset_item(split, int(idx) - 1)
            return img, new_idx, info

        def on_next(split, idx):
            img, new_idx, info = _dataset_item(split, int(idx) + 1)
            return img, new_idx, info

        def on_upload(img):
            if img is None:
                return gr.skip()
            return img

        split_radio.change(
            fn=on_split_change,
            inputs=[split_radio],
            outputs=[display_img, idx_state, idx_info],
        )
        prev_btn.click(
            fn=on_prev,
            inputs=[split_radio, idx_state],
            outputs=[display_img, idx_state, idx_info],
        )
        next_btn.click(
            fn=on_next,
            inputs=[split_radio, idx_state],
            outputs=[display_img, idx_state, idx_info],
        )
        upload_img.change(
            fn=on_upload,
            inputs=[upload_img],
            outputs=[display_img],
        )
        classify_btn.click(
            fn=classify,
            inputs=[display_img],
            outputs=[result_md],
        )

        # load initial image on page open
        demo.load(
            fn=lambda: _dataset_item("test", 0),
            outputs=[display_img, idx_state, idx_info],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
