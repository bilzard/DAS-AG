# Image Generation with Aesthetic Guidance

This project introduces **Direct Ascent Synthesis with Aesthetic Guidance (DAS-AG)**, an extension of **Direct Ascent Synthesis (DAS)** [1]. DAS is a method that generates images guided by **CLIP loss**, optimizing images to match a given text prompt. DAS-AG enhances this approach by incorporating an **Aesthetic Model's predicted score** as an auxiliary loss. By leveraging this aesthetic score, we can control the aesthetics of the generated images:

1. **Adding the aesthetic score as a reward** leads to more aesthetically pleasing results compared to standard DAS.
2. **Using the aesthetic score as a penalty** (i.e., applying a negative weight) results in less aesthetically pleasing images.

## Differences from the Authors' Original Code

I implemented this repository independently based on the author's paper, as I was not aware of the authors' original implementation [2].
As a result, some implementation details differ, including:

- **Resolution selection**:
  - My implementation uses only **powers of 2** (e.g., [1, 2, 4, 8, ...]) to reduce computation.
  - The original implementation considers **all resolutions** (e.g., [1, 2, 3, 4, ...]).
- **Initial noise setting**:
  - The initial image noise is set to **1/size** in my implementation.
  - This adjustment, based on preliminary experiments, tends to produce **less noisy results**.

## Prerequisites

- [uv](https://docs.astral.sh/uv/concepts/tools/)

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

### Download pre-trained model weights

#### Downloading the Pre-trained CLIP Model

You can download it with this python code.
```python
import clip
_ = clip.load('ViT-L/14', jit=True, download_root="/path/to/local")
```
#### Downloading CLIP+MLP Aesthetic Score Predictor

You can download it from the author's Github repository.

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth /path/to/local
```

#### Update Environment Variables

Make sure to update `.env` with the correct path:

```bash
CLIP_MODEL_PATH="/path/to/clip/ViT-L-14.pt"
AESTHETIC_PREDICTOR_PATH="/path/to/kaggle-dwllm/sac+logos+ava1-l14-linearMSE.pth"
```

## Execute Application

### Running the Demo

To run the demo, execute the following command:
```bash
uv run streamlit run src/das_ag/app.py
```

### Reproducible Execution

If you need **deterministic results**, use one of the following commands instead.
This ensures consistent computations for training but may reduce efficiency.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run streamlit run src/das_ag/app.py
# or
CUBLAS_WORKSPACE_CONFIG=:16:8 uv run streamlit run src/das_ag/app.py
```

For more details on reproducibility, see the [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).

## Results

For training, I use **CLIP+MLP Aesthetic Score Predictor** [4] and evaluate the results with **Aesthetic Predictor v2.5** [3].
To prevent overfitting, I use different aesthetic prediction models for training and evaluation, with distinct backbones and heads.

The table below presents the evaluated aesthetic scores for different generation methods.
Each score represents the mean and standard deviation computed from **six samples per method**, including different prompts.

| Method  | Mean  | Std  |
|---------|------|------|
| **DAS**     | 3.406 | 0.230 |
| **DAS-AG**  | **3.783** | 0.203 |
| **DAS-rAG** | 2.988 | 0.282 |

### **Generation Methods**
- **DAS**: Direct Ascent Synthesis [1]
- **DAS-AG**: DAS with aesthetic guidance
- **DAS-rAG**: DAS with reverse aesthetic guidance

For detailed experimental settings, see **Appendix A** in `note/experimental_details.md`.

## Example Generated Images

Here, I provide examples of generated images, along with their aesthetic scores.
I also provide qualitative analysis of these images in `note/experimental_details.md`.

### Example 1: "Gigantic Mona Lisa Attacks the City"

<table>
    <tr>
        <th><b>Method</b></th>
        <th>DAS-rAG</th>
        <th>DAS</th>
        <th>DAS-AG</th>
    </tr>
    <tr>
        <td><b>Aesthetic Score (↑ better)</b></td>
        <td>2.69</td>
        <td>2.83</td>
        <td>3.47</td>
    </tr>
    <tr>
        <td><b>Generated Image</b></td>
        <td>
            <img src="https://github.com/user-attachments/assets/909b503c-045e-4c92-9f25-11960dddcf4a"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/0c82d3fd-43b7-48e0-ae4b-d0a6fa15c384"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/f78b57da-15c3-4e0d-9e2f-742ff71ff675">
        </td>
    </tr>
</table>

### Example 2: "Japanese Geisha Girl"

<table>
    <tr>
        <th><b>Method</b></th>
        <th>DAS-rAG</th>
        <th>DAS</th>
        <th>DAS-AG</th>
    </tr>
    <tr>
        <td><b>Aesthetic Score (↑ better)</b></td>
        <td>2.66</td>
        <td>3.31</td>
        <td>4.06</td>
    </tr>
    <tr>
        <td><b>Generated Image</b></td>
        <td>
            <img src="https://github.com/user-attachments/assets/f6efa726-f4e3-4ad5-af61-1432ba725506"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/a972adba-ecda-4d89-95d3-0de6b772ed75"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/f51b88fe-94d0-4b0d-8ce9-0e25bd64a718">
        </td>
    </tr>
</table>

### Example 3: "Gamma-ray Bursts Over Tokyo Tower"

<table>
    <tr>
        <th><b>Method</b></th>
        <th>DAS-rAG</th>
        <th>DAS</th>
        <th>DAS-AG</th>
    </tr>
    <tr>
        <td><b>Aesthetic Score (↑ better)</b></td>
        <td>3.00</td>
        <td>3.80</td>
        <td>3.62</td>
    </tr>
    <tr>
        <td><b>Generated Image</b></td>
        <td>
            <img src="https://github.com/user-attachments/assets/dfb3318f-1da1-4730-98c9-1a4439700a21"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/1a6d5cb9-54b5-4619-a009-782a1ea12868"/>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/7a82e1c3-fbd5-4d1b-b04b-a27155213a74">
        </td>
    </tr>
</table>

## Acknowledgements

This project builds upon the work presented in *Direct Ascent Synthesis* (DAS) [1]. I would like to thank to the authors for their contributions.

Additionally, this project makes use of the following publicly available tools. I also would like to thank the authors for their work:

- **OpenAI**: [CLIP](https://github.com/openai/CLIP) (used for feature extraction, licensed under Apache 2.0)
- **@discus0434**: [Aesthetic Predictor V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5) (used for evaluation, licensed under AGPL-3.0)
- **Christoph Schuhmann**: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (licensed under Apache 2.0)

## License

This project is licensed under the Apache License 2.0.

This project includes code modified from:
- [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (licensed under Apache 2.0)

For details about third-party code and modifications, see the [`NOTICE`](NOTICE) file and the license files in the [`licenses/`](licenses/) directory.

## References

- [1] Direct Ascent Synthesis: Revealing Hidden Generative Capabilities in Discriminative Models, 11 Feb 2025, https://arxiv.org/abs/2502.07753
- [2] https://github.com/stanislavfort/Direct_Ascent_Synthesis
- [3] Aesthetic Predictor v2.5, https://github.com/discus0434/aesthetic-predictor-v2-5
- [4] CLIP+MLP Aesthetic Score Predictor, https://github.com/christophschuhmann/improved-aesthetic-predictor?tab=readme-ov-file
