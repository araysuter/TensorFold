# Arete 5-bit and native vision qualification

The [summary](validation.json) and [raw index](raw-index.json) describe the
whitelisted JSON files in `raw-results.tar.gz`. Each member has a SHA256 and
byte count. The archive contains public synthetic request hashes, token hashes,
numeric receipts and validation results; no credentials, private configuration,
snapshot payloads, model weights or private logs are included.

The successful Arete run has 26 checks, including a 130,944-token cold prompt
and SSD-restored continuation reusing 130,950 tokens. Its native lifetime peak
physical footprint was 45,256,262,096 bytes. Swift and Ornith each passed 19
image/lifecycle checks. Their new image runs are short-context checks, not new
128K vision capacity measurements. All profiles configured a 131,072-token
window. Standalone MLX-VLM correctly answered the two Arete color fixtures.

The first Arete attempt retained its four successful plain/drafted text pairs
but failed a fixture that resent the previous request instead of a true
continuation. The corrected text continuation passed. The next attempt exposed
an unsupported image-history preflight: image follow-ups run cold. The final
image gate checks fresh-output equality and records cache reuse as false;
image SSD reuse is not advertised. Both failed receipts are retained.

The first Ornith startup attempt returned HTTP 503 while the memory watchdog
was active; its exact stop reason was not retained. A fresh cold start passed
all 19 checks under a 6 GiB host reserve. No tolerance or token-identity gate
was widened. The benchmark tool now records a watchdog stop's telemetry.

Calibration used isolated loopback gateways with profile budgets temporarily
disabled and explicit physical-memory/pressure watchdogs. The default serving
configuration enables per-profile admission plus the 6 GiB host reserve. A
calibration pass is not a promise that the same model can be admitted alongside
an arbitrary training job. No new quality score, general vision-accuracy score,
kernel speedup or other-platform qualification is claimed.

The [converter](../../tools/arete/convert.py),
[BF16 tower attachment](../../tools/prepare-qwen-vision.py), and
[opt-in probe](../../tools/arete/probe.py) reproduce the operations. Native
session support uses the existing pinned build described in
[the native package](../../tools/studio-gateway/native/README.md).
