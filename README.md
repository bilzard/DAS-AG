# (Project Name)

This repository contains the resources required to reproduce submissions for [(Competition Name)]().

## Prerequisites

- [uv](https://docs.astral.sh/uv/concepts/tools/)
- NVIDIA Ampere or newer GPUs (for FlashAttention v2)

## Install

### Install Required Packages

This project requires the `flash-attn` package, which has specific build dependencies.
To ensure proper installation, we need to run `uv sync` twice:

1. Install standard dependencies:
    ```bash
    uv sync --extra build
    ```
2. Compile and install flash-attn:
    ```bash
    uv sync --extra build --extra compile
    ```

For more details, please refer to [official documentation of uv](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation).

**Note**: If you want to add a new dependency to the `build` optional dependencies, use the following command:
```bash
uv add --optional build torchvision
```

## How to Reproduce

## Result

## Acknowledgements

## Reference
