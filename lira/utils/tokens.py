# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.


def load_tokens(token_path):
    with open(token_path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f]
