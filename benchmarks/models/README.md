# Benchmark Models

## MCUNet Models (MIT HAN Lab)

10 int8 quantized TFLite models from the MCUNet model zoo.

Source: https://github.com/mit-han-lab/mcunet
Paper: "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)

### Download

```bash
# Download all 10 MCUNet models (with SHA256 verification)
./benchmarks/models/download_mcunet.sh

# Verify existing files only
./benchmarks/models/download_mcunet.sh --verify

# Force re-download
./benchmarks/models/download_mcunet.sh --force
```

Or manually via Python:

```python
from mcunet.model_zoo import download_tflite
tflite_path = download_tflite(net_id="mcunet-in0")
```

### Name Mapping

The tflite filenames use MCUNet's official `net_name` from `model_zoo.py`.
When compiled with te-comp, hyphens and dots are replaced with underscores
to form valid C identifiers:

| MCUNet net_id | tflite filename | C identifier (output dir) |
|---------------|-----------------|---------------------------|
| mcunet-in0 | `mcunet-10fps_imagenet.tflite` | `mcunet_10fps_imagenet` |
| mcunet-in1 | `mcunet-5fps_imagenet.tflite` | `mcunet_5fps_imagenet` |
| mcunet-in2 | `mcunet-256kb-1mb_imagenet.tflite` | `mcunet_256kb_1mb_imagenet` |
| mcunet-in3 | `mcunet-320kb-1mb_imagenet.tflite` | `mcunet_320kb_1mb_imagenet` |
| mcunet-in4 | `mcunet-512kb-2mb_imagenet.tflite` | `mcunet_512kb_2mb_imagenet` |
| mbv2-w0.35 | `mbv2-w0.35-r144_imagenet.tflite` | `mbv2_w0_35_r144_imagenet` |
| proxyless-w0.3 | `proxyless-w0.3-r176_imagenet.tflite` | `proxyless_w0_3_r176_imagenet` |
| mcunet-vww0 | `mcunet-10fps_vww.tflite` | `mcunet_10fps_vww` |
| mcunet-vww1 | `mcunet-5fps_vww.tflite` | `mcunet_5fps_vww` |
| mcunet-vww2 | `mcunet-320kb-1mb_vww.tflite` | `mcunet_320kb_1mb_vww` |

### Download URL Pattern

```
https://hanlab18.mit.edu/projects/tinyml/mcunet/release/{net_name}.tflite
```

where `net_name` is from the `NET_INFO` dict in
[mcunet/model_zoo.py](https://github.com/mit-han-lab/mcunet/blob/main/mcunet/model_zoo.py).
