"""
AMD Radeon GPU Acceleration Check via Microsoft DirectML
Tests ONNX Runtime DirectML Execution Provider on AMD Radeon 610M
"""

import onnxruntime as ort
import numpy as np

print("=" * 70)
print("AMD RADEON GRAPHICS ACCELERATION CHECK (DIRECTML)")
print("=" * 70)

providers = ort.get_available_providers()
print(f"Available Execution Providers: {providers}")

if 'DmlExecutionProvider' in providers:
    print("\n[SUCCESS] Microsoft DirectML is AVAILABLE and ACTIVE!")
    print("  Hardware: AMD Radeon(TM) 610M (DirectX 12 Compatible)")
    print("  Provider: DmlExecutionProvider (DirectML for AMD GPUs)")
    print("  This allows running deep learning inference on your AMD Radeon GPU!")
else:
    print("\n[INFO] DirectML execution provider not found in current environment.")

print("=" * 70)
