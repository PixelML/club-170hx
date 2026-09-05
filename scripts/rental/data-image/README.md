# specdec-sliceb data image

Reusable data-transport image for PixelML/club-170hx GLM-5.3-Flash drafter
training rentals. Rental instances without a docker binary pull it via the
raw GHCR registry API instead of `docker pull` — see `pull_data_image.py`.

## Contents (/data/)

- sliceB/ - 9 shards (shard-0000.npz .. shard-0008.npz) + manifest.json +
  reference-row.json + SHA256SUMS. Hidden-state slice-B training data,
  tap hc_post-materialized+stream-mean. ~14 GB.
- target-shared.safetensors - shared target-model tensors (embed_tokens +
  lm_head) used to initialize/verify drafter training. ~2.5 GB.
- specdec-tools-e03679f1.tar.gz - pinned tools tarball, commit
  e03679f119af5ce21c8e83f2e267166add79c8f7 (exported-checkpoint-set-only
  variant, ~9.4 GB/run instead of ~19 GB/run - supersedes the earlier
  eb1434a88ddd292079b6087b374f2bacb64ba816 pin).
- CHECKSUMS.sha256 - sha256sum of every file above (sha256sum -c).

## Provenance / licensing

- Target model: GLM-5.3-Flash (AWQ W4A16 quant, wtdcode/GLM-5.3-Flash-AWQ-W4A16
  on HF) - MIT.
- Training corpora tapped into sliceB: ultrachat_200k (MIT), tulu-3 (ODC-BY).
- Tap: hc_post-materialized+stream-mean.

## NOT included

The reference drafter (incoai/GLM-5.3-Flash-DFlash2) is CC-BY-NC-ND and must
not be redistributed. Rental instances pull it directly from HuggingFace
with HF_TOKEN set as an environment variable - do not bake a token or the
weights into this image.

## Pulling on a rental instance (no docker daemon available)

    python3 pull_data_image.py \
      --image ghcr.io/pixelml/club-170hx \
      --tag specdec-sliceb-20260905 \
      --dest /data
    sha256sum -c /data/CHECKSUMS.sha256
