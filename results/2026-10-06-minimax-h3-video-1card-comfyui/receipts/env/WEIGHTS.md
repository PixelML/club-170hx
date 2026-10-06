**Weights used (SHA-256 of the served files, matched to pinned HF LFS hashes)**

| local file | bytes | sha256 | matches (repo @ rev : path) |
|---|---|---|---|
| `<models>/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67…` | Comfy-Org/MiniMax-H3 @ e5eb578a : diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors |
| `<models>/h3/loras/minimax_h3_turbo_4step_ema_ckpt850.safetensors` | 779,849,816 | `5a6eeba171cf1830…` | larryvrh/MiniMax-H3-Turbo-Lora @ 43a74557 : minimax_h3_turbo_4step_ema_ckpt850.safetensors |
| `<models>/h3/loras/minimax_h3_turbo_v4_step600_ema.safetensors` | 779,849,816 | `5f3a626cd72c93a8…` | larryvrh/MiniMax-H3-Turbo-Lora @ 43a74557 : minimax_h3_turbo_v4_step600_ema.safetensors |
| `<models>/h3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `9255f52b6677845a…` | Comfy-Org/MiniMax-H3 @ e5eb578a : diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors |
| `<models>/h3/vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d4…` | Comfy-Org/MiniMax-H3 @ e5eb578a : vae/minimax_h3_audio_vae_fp32.safetensors |
| `<models>/h3/vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7edda…` | Comfy-Org/MiniMax-H3 @ e5eb578a : vae/minimax_h3_video_vae_fp16.safetensors |
| `<models>/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | `bc2ced0fbea64757…` | Comfy-Org/MiniMax-H3 @ e5eb578a : text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors |

Official checkpoint for SGLang/diffusers: `<models>/MiniMax-H3` = MiniMaxAI/MiniMax-H3 @ 42ed227e (Ref2VA/, transformer_ref/, vae/, audio_vae/; text_encoder/ is a symlink to Ref2VA/text_encoder, same LFS blobs).
