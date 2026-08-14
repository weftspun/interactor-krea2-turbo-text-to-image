"""Krea 2 Turbo text to image. RFD 0042.

Four models live in one folder: a 12.0 B backbone, a 4.7 B T5, a
0.12 B CLIP, and a 0.08 B VAE. The loads stage: the text encoders run
and unload before the backbone loads, since the backbone never reads
the encoders again. That keeps the peak near the backbone size, and
not near the sum. RFD 0036's "setup() loads, predict() loads nothing"
rule is bent on purpose here (per RFD 0042); the staging is the
reason -- load() maps every file, predict() moves parts to the device
per stage.

Ships Q4_K_M only in this image (9.30 GB of the 33.8 GB bf16 set) --
RFD 0042 flags a 57 GB weight folder trap: the folder carries fp32
copies too, and a build that copies the whole folder makes a 57 GB
image for a 9.30 GB model. The Dockerfile fetches only the four
Q4_K_M GGUF files by name, never the folder.
"""

import base64
import os
import tempfile
from pathlib import Path

STUB = os.environ.get("WEFTSPUN_STUB") == "1"
_READY = {"loaded": False}

# Turbo. More steps cost time and give no better image (RFD 0042).
DEFAULT_STEPS = 4


class InputError(ValueError):
    """The request is wrong. This is the caller's fault, and not ours."""


def _validate(job_input: dict) -> dict:
    if not job_input.get("prompt"):
        raise InputError("prompt is required")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    if not (256 <= width <= 2048) or not (256 <= height <= 2048):
        raise InputError("width and height must be between 256 and 2048")
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    if not (1 <= steps <= 12):
        raise InputError("steps must be between 1 and 12")
    return {
        "prompt": job_input["prompt"],
        "negative_prompt": job_input.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "steps": steps,
        "seed": int(job_input.get("seed", -1)),
    }


def _run_upstream(args: dict, work: Path) -> Path:
    """Stage 1: encode with T5+CLIP, then free them. Stage 2: denoise
    with the backbone. Stage 3: decode with the VAE. Not yet wired --
    the staged Krea 2 Turbo pipeline call against the Q4_K_M GGUF set
    is the documented pattern for this format, but wasn't executed
    against the real package in this pass."""
    raise NotImplementedError(
        "Port the staged Krea 2 Turbo pipeline (T5/CLIP encode -> backbone "
        "denoise -> VAE decode) here -- see README's Status"
    )


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict(job_input: dict) -> dict:
    args = _validate(job_input)
    work = Path(tempfile.mkdtemp())

    if STUB:
        image = work / "stub.png"
        image.write_bytes(bytes([0x89, 0x50, 0x4E, 0x47]) + b"stub")
    else:
        image = _run_upstream(args, work)

    return {
        "image": _encode(image),
        "seed": args["seed"],
        "stub": STUB,
    }


def load() -> None:
    _READY["loaded"] = True


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="krea2_turbo_text_to_image", version="0.1.0")

    class PredictRequest(BaseModel):
        prompt: str
        negative_prompt: str = ""
        width: int = 1024
        height: int = 1024
        steps: int = DEFAULT_STEPS
        seed: int = -1

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
