# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility re-export of SM80 `moe_align_block_size` from `sm80_ops_v2`."""

from .sm80_ops_v2 import moe_align_block_size

__all__ = ["moe_align_block_size"]
