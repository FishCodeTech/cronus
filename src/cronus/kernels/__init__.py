"""Audited FLA Triton Autotuner pinning utilities for SM_121 / GB10."""

from cronus.kernels.native_pin import install as install_native_pin
from cronus.kernels.train_pin import install as install_train_pin

__all__ = ["install_native_pin", "install_train_pin"]
