# Native image profiles

Swift, Ornith and Arete declare compatible Qwen vision towers. Ripple is a
text-only model with no trained image encoder; its text-only request validation
remains enabled. Adding an unrelated encoder would not give it trained vision.

Keep the existing language quantization and use the model's original BF16 vision
tower where available. A tower is about 0.92 GB; images also need activation and
scratch memory. Include those costs in each profile's memory budget. Global
conversation retention remains four total, with one resident model and drafter.

| Profile | Language weights | Vision weights |
|---|---|---|
| Swift | Existing native-prepared 4-bit weights | Original Swift BF16 tower, separate preparation below |
| Ornith mixed 2-bit | Existing published quant | Already includes 333 BF16 vision tensors and processor assets |
| Arete | MLX affine 5-bit, group 64 | Original Arete BF16 tower; see [conversion](../arete/README.md) |
| Ripple | Existing mixed 4-bit | None; image input is refused |

All three published Ornith quants retain the BF16 tower. The gateway uses the
smallest quant. The other two are not downloaded or qualified again just to
enable image input. Their larger language weights require separate capacity
checks before deploying them on a 64 GiB host.

## Prepare Swift without changing the active checkpoint

The original Swift MLX snapshot has a quantized `visual.*` tower. The native
loader reads a BF16 `vision_tower.*` tower. Keep its already prepared language
weights unchanged and attach the original Swift vision tensors in a separate
directory. Do not apply the language RMSNorm conversion a second time.

Verify these destinations are excluded from Time Machine before large downloads
on a backed-up Mac. The source download selects only the approximately 4 GB shard
containing vision and its small metadata, not the full BF16 language model:

```sh
hf download ukisai/Swift-1.5-Qwen3.8-27b \
  --revision b4c84d42903a8646b25857eb2827288d92ed85a4 \
  --include 'config.json' 'model.safetensors.index.json' \
    'model-00001-of-00018.safetensors' '*processor*.json' 'LICENSE' \
  --local-dir "$HOME/ai/models/swift-vision-source-b4c84d42"
python3 - <<'PY'
from pathlib import Path
import os
source = Path.home() / 'ai/models/swift-1.5-native-82276731'
target = Path.home() / 'ai/models/swift-1.5-native-vision-b4c84d42'
target.mkdir(exist_ok=False)
for path in source.iterdir():
    if path.is_file():
        if path.suffix == '.safetensors':
            os.link(path, target / path.name)
        else:
            (target / path.name).write_bytes(path.read_bytes())
PY
python3 tools/prepare-qwen-vision.py \
  "$HOME/ai/models/swift-vision-source-b4c84d42" \
  "$HOME/ai/models/swift-1.5-native-vision-b4c84d42" \
  --source-repo ukisai/Swift-1.5-Qwen3.8-27b \
  --source-revision b4c84d42903a8646b25857eb2827288d92ed85a4 \
  --drop-legacy-visual-index
```

Hardlinked language shard files must remain immutable in both directories. Only
the new copy's index and small configuration files change. Legacy quantized
vision tensors remain in their old shard but are removed from the native index;
this is a native preparation, not a replacement MLX-VLM export of Swift.

## API input and limits

Add `--vision --vision-image-tokens 2048 --vision-max-images 2` to the native
profile. Bound video scratch too with `--vision-video-tokens 2048
--vision-max-videos 1` where those options are supported. Image tests do not
qualify video input. Keep public URL fetching disabled; send images as inline
data URLs. The gateway's request size and the native image-token limit still
apply, and image tokens share the model's total context window.

```json
{
  "model": "arete-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ENCODED_PNG"}}
  ]}],
  "max_tokens": 256
}
```

Text continuations retain their tested RAM and SSD reuse. The current native
image-prefix preflight reports a miss for image histories, so image follow-ups
run cold; do not advertise image SSD reuse. Changed-image and fresh-run output
checks are required before exposing a vision profile. A cold reply is not a
claimed cache hit. Memory admission checks remain enabled for production.
