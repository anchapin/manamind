# seed0_settle iter 45, schema simple-v1

The checkpoint behind the four difficulty tiers (#46). See `docs/embedding.md` for the host contract.

- Source: `seed0_settle/checkpoints/iter_045.pt` on the home PC (sha256 in `source_checkpoint.sha256`)
- Exported by run 37525539204 (`export-onnx.yml`, main `dedd8ba`)
- Parity vs PyTorch on 64 positions: max |logit diff| 3.81e-06, max |value diff| 2.24e-08
- Reference host (`scripts/onnx_host.py`, policy only, temperature 1.0, ONNX only): 28/40 vs RandomAgent (.700). The 160-game Easy tier figure is .775; this result is within the 40-game interval.
