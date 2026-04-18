import logging
from functools import lru_cache
from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from PIL import Image

from checkpoint import restore_checkpoint
from configs.default import get_config
from models import forward, init_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


def list_runs() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(p.name for p in CHECKPOINT_DIR.iterdir() if p.is_dir())


def list_epochs(run_name: str | None) -> list[str]:
    if not run_name:
        return []
    run_dir = CHECKPOINT_DIR / run_name
    if not run_dir.exists():
        return []
    return sorted(
        p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")
    )


@lru_cache(maxsize=8)
def load_params(run_name: str, epoch_name: str) -> dict:
    config = get_config()
    key = jax.random.PRNGKey(config.seed)
    params_template = init_params(
        key,
        input_dim=784,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    )
    optimizer = optax.sgd(config.learning_rate, momentum=config.momentum)
    opt_state_template = optimizer.init(params_template)

    epoch_num = int(epoch_name.split("_")[-1])
    run_dir = str((CHECKPOINT_DIR / run_name).resolve())
    checkpointer = ocp.StandardCheckpointer()
    params, _, _ = restore_checkpoint(
        checkpointer, run_dir, epoch_num, params_template, opt_state_template
    )
    logger.info("Loaded checkpoint: %s / %s", run_name, epoch_name)
    return params


def preprocess_sketch(sketch) -> jax.Array | None:
    if sketch is None:
        return None
    img = sketch["composite"] if isinstance(sketch, dict) else sketch
    if img is None:
        return None

    pil = Image.fromarray(np.asarray(img)).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0

    # MNIST is white-on-black; auto-invert if the image is mostly light.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return jnp.array(arr.reshape(1, 784))


def predict(sketch, run_name: str, epoch_name: str) -> dict[str, float]:
    if not run_name or not epoch_name:
        return {}
    x = preprocess_sketch(sketch)
    if x is None:
        return {}
    params = load_params(run_name, epoch_name)
    probs = forward(params, x)[0]
    return {str(i): float(probs[i]) for i in range(10)}


def build_app() -> gr.Blocks:
    runs = list_runs()
    initial_run = runs[0] if runs else None
    epochs = list_epochs(initial_run)
    initial_epoch = epochs[-1] if epochs else None

    with gr.Blocks(title="MNIST Sketch Inference") as demo:
        gr.Markdown("# MNIST Sketch Inference\nDraw a digit (0–9) and select a checkpoint.")
        with gr.Row():
            with gr.Column():
                run_dd = gr.Dropdown(choices=runs, value=initial_run, label="Run")
                epoch_dd = gr.Dropdown(
                    choices=epochs, value=initial_epoch, label="Epoch"
                )
                sketch = gr.Sketchpad(
                    label="Draw a digit",
                    type="numpy",
                    image_mode="L",
                    height=320,
                    width=320,
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                output = gr.Label(label="Prediction", num_top_classes=10)

        def on_run_change(r: str):
            eps = list_epochs(r)
            return gr.update(choices=eps, value=eps[-1] if eps else None)

        run_dd.change(on_run_change, inputs=[run_dd], outputs=[epoch_dd])
        predict_btn.click(
            predict, inputs=[sketch, run_dd, epoch_dd], outputs=[output]
        )

    return demo


def main() -> None:
    build_app().launch(share=True)


if __name__ == "__main__":
    main()
