# interactor-krea2-turbo-text-to-image

Model image for `krea2_turbo_text_to_image`, per
[weftspun's RFD 0036](https://github.com/weftspun/request-for-discussion/tree/main/0036-packaging-convention)
packaging convention. Facts from
[RFD 0042](https://github.com/weftspun/request-for-discussion/tree/main/0042-krea2-turbo-text-to-image).

## Model

| Property | Value |
|---|---|
| Upstream | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Raw) (Krea AI), 4 parts: backbone (12.0 B), T5 (4.7 B), CLIP (0.12 B), VAE (0.08 B) |
| License | **Krea 2 Community License** (custom, not Apache/MIT) — free commercial use requires company-wide annual revenue under $1M **and** fewer than 50 seats; larger orgs need a separate enterprise license. Independently checked against [krea.ai/krea-2-licensing](https://www.krea.ai/krea-2-licensing); this is revenue-gated, not a blanket commercial grant — flagging for whoever owns RFD 0028's license gate to confirm this clears the bar for every deployer, not just weftspun itself. |
| Parameters | 16.9 B total, published |
| bf16 | 33.8 GB — largest model in the catalog |
| Q4_K_M | 9.30 GB — the only ship format, the largest single saving in the catalog |

## The disk trap (RFD 0042)

The real weight folder on HF is ~57 GB — it carries fp32 copies alongside the bf16 set. A build
that clones or snapshot-downloads the whole folder makes a 57 GB image for a 9.30 GB model. This
Dockerfile fetches the four named Q4_K_M GGUF files individually instead.

**The exact HF repo id and filenames for the Q4_K_M GGUF set are not yet independently confirmed**
— `KREA2_REPO` in the Dockerfile is a best-guess based on Krea's naming pattern for other releases,
not verified against a real listing. Confirm before trusting the worker stage's weight fetch.

## The staged load (also RFD 0042)

Four models live in one folder. The text encoders (T5, CLIP) run and unload before the backbone
loads, since the backbone never reads the encoders again — this keeps the peak VRAM near the
backbone's size, not the sum of all four. RFD 0036's usual "setup() loads weights, predict() loads
nothing" rule is bent on purpose here, per RFD 0042: `load()` maps every file, `predict()` moves
parts to the device per stage.

## Interface

`POST /predict`:

| Input | Type | Default | Note |
|---|---|---|---|
| `prompt` | str | required | |
| `negative_prompt` | str | "" | |
| `width` | int | 1024 | 256–2048 |
| `height` | int | 1024 | 256–2048 |
| `steps` | int | 4 | 1–12; this is the Turbo variant — 30 steps costs 7× and gives no better image |
| `seed` | int | -1 | |

Returns `{image, seed, stub}`.

## Build

```sh
docker build --target contract -t interactor-krea2-turbo-text-to-image:contract .
docker run --rm -p 8000:8000 interactor-krea2-turbo-text-to-image:contract
curl -X POST localhost:8000/predict -d @test_input.json -H 'Content-Type: application/json'
```

## Status

**Scaffolded from the RFD, not yet built or run.** `_run_upstream()` raises `NotImplementedError`
by design — the staged encode/denoise/decode pipeline against the Q4_K_M GGUF set is the
documented shape for this format, but wasn't executed against the real package in this pass.
Confirm the exact HF repo id/filenames and the diffusers-or-custom pipeline class before trusting
the worker stage.
