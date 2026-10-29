#!/usr/bin/env python3
"""
Create 6 quantised int8 TFLite benchmark models.

Models:
  1. MNIST CNN         - digit classification (Conv→Pool→FC)
  2. CIFAR-10 CNN      - image classification (3x Conv→Pool→FC)
  3. MobileNetV1-0.25  - depthwise separable convolutions
  4. Keyword Spotting   - DS-CNN for audio (from MLPerf Tiny spec)
  5. FC Autoencoder     - anomaly detection (FC-only)
  6. Person Detection   - visual wake word (MobileNet-based)

All models are post-training quantised to int8 using representative
datasets (random data for benchmarking purposes).

Usage:
  python3 create_models.py [output_dir]
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Workaround for broken system pyopenssl on Ubuntu 20.04
try:
    import OpenSSL  # noqa: F401
except (ImportError, AttributeError):
    sys.modules['OpenSSL'] = type(sys)('OpenSSL')
    sys.modules['OpenSSL.SSL'] = type(sys)('OpenSSL.SSL')
    sys.modules['OpenSSL.crypto'] = type(sys)('OpenSSL.crypto')

import tensorflow as tf


def representative_dataset_gen(shape, n_samples=100):
    """Generate representative data for quantisation calibration."""
    def gen():
        for _ in range(n_samples):
            yield [np.random.uniform(-1, 1, shape).astype(np.float32)]
    return gen


def quantise_model(model, input_shape):
    """Convert Keras model to int8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(
        [1] + list(input_shape))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def get_model_info(tflite_path):
    """Get model size and op info from TFLite file."""
    size = os.path.getsize(tflite_path)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    # Count ops
    # Use the _get_ops_details if available
    try:
        ops = interpreter._get_ops_details()
        op_names = [op['op_name'] for op in ops]
    except Exception:
        op_names = ['(unknown)']
    return {
        'size': size,
        'input_shape': inp[0]['shape'].tolist(),
        'input_dtype': str(inp[0]['dtype']),
        'output_shape': out[0]['shape'].tolist(),
        'output_dtype': str(out[0]['dtype']),
        'ops': op_names,
        'n_ops': len(op_names),
    }


# ============================================================================
# Model 1: MNIST CNN
# ============================================================================

def create_mnist_cnn():
    """Conv2D(8,3x3) → ReLU → MaxPool(2x2) → Conv2D(16,3x3) → ReLU →
    MaxPool(2x2) → Flatten → FC(10)"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
    ])
    model.build()
    return model, (28, 28, 1)


# ============================================================================
# Model 2: CIFAR-10 CNN
# ============================================================================

def create_cifar10_cnn():
    """3-layer CNN for CIFAR-10 (32x32x3 input)."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model.build()
    return model, (32, 32, 3)


# ============================================================================
# Model 3: MobileNetV1-0.25 (128x128)
# ============================================================================

def create_mobilenetv1_025():
    """MobileNetV1 with alpha=0.25, 128x128 input.
    Uses depthwise separable convolutions."""
    model = tf.keras.applications.MobileNet(
        input_shape=(128, 128, 3),
        alpha=0.25,
        depth_multiplier=1,
        include_top=True,
        weights=None,
        classes=10,
    )
    return model, (128, 128, 3)


# ============================================================================
# Model 4: Keyword Spotting DS-CNN (MLPerf Tiny)
# ============================================================================

def create_kws_ds_cnn():
    """Depthwise separable CNN for keyword spotting.
    Input: 49x10 MFCC features (reshaped to 49x10x1).
    Based on MLPerf Tiny DS-CNN-S architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(49, 10, 1)),
        # First regular conv
        tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        # DS-Conv blocks
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                         activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                         activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                         activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # Global average pooling + classifier
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(12),  # 12 keyword classes
    ])
    model.build()
    return model, (49, 10, 1)


# ============================================================================
# Model 5: FC Autoencoder (Anomaly Detection)
# ============================================================================

def create_fc_autoencoder():
    """Fully connected autoencoder for vibration anomaly detection.
    Based on MLPerf Tiny anomaly detection spec.
    Input: 640-dim feature vector (5 frames x 128 MFCC bins)."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(640,)),
        # Encoder
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),     # bottleneck
        # Decoder
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(640),                      # reconstruction
    ])
    model.build()
    return model, (640,)


# ============================================================================
# Model 6: Person Detection (Visual Wake Word)
# ============================================================================

def create_person_detection():
    """MobileNetV1-0.25 for person detection (96x96 input).
    Based on MLPerf Tiny visual wake word benchmark."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
        # Use MobileNetV1 backbone
        tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2),
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        # DS-Conv blocks (simplified MobileNet)
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same',
                                         activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2),
                                         padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2),
                                         padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2),
                                         padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # Classifier
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2),  # person / not-person
    ])
    model.build()
    return model, (96, 96, 3)


# ============================================================================
# Main
# ============================================================================

MODELS = [
    ('01_mnist_cnn', create_mnist_cnn),
    ('02_cifar10_cnn', create_cifar10_cnn),
    ('03_mobilenet_v1_025', create_mobilenetv1_025),
    ('04_kws_ds_cnn', create_kws_ds_cnn),
    ('05_fc_autoencoder', create_fc_autoencoder),
    ('06_person_detection', create_person_detection),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'models'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Creating Benchmark Models")
    print("=" * 70)

    results = []

    for name, create_fn in MODELS:
        print(f"\n--- {name} ---")
        try:
            model, input_shape = create_fn()
            n_params = model.count_params()
            print(f"  Params: {n_params:,}")
            print(f"  Input:  {input_shape}")

            # Quantise to int8 TFLite
            tflite_data = quantise_model(model, input_shape)
            tflite_path = os.path.join(out_dir, f'{name}.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_data)

            info = get_model_info(tflite_path)
            print(f"  TFLite: {info['size']:,} bytes ({info['size']/1024:.1f} KB)")
            print(f"  In:     {info['input_shape']} ({info['input_dtype']})")
            print(f"  Out:    {info['output_shape']} ({info['output_dtype']})")
            print(f"  Ops:    {info['n_ops']} ({', '.join(set(info['ops']))})")

            results.append({
                'name': name,
                'params': n_params,
                'tflite_size': info['size'],
                'input_shape': info['input_shape'],
                'output_shape': info['output_shape'],
                'n_ops': info['n_ops'],
                'ops': list(set(info['ops'])),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Model':<25} {'Params':>10} {'TFLite':>10} {'Ops':>5}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>10,} "
              f"{r['tflite_size']:>9,}B {r['n_ops']:>5}")

    # Save summary as JSON
    import json
    summary_path = os.path.join(out_dir, 'model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
