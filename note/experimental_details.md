# Experimental Details

## Qualitative Analysis

Here, I provide qualitative analysis of the generated images in "Example Generated Images" section in `README.md`.

The exact prompts used in these examples are provided in **Appendix B**.

### Example 1: "Gigantic Mona Lisa Attacks the City"

This example illustrates a successful case of the proposed method (DAS-AG and DAS-rAG). The result of DAS-rAG is more detailed— for example, both the clothing textures of the person in the front and the back-standing Mona Lisa are more refined. Additionally, the perspective appears more accurate. In contrast, the result of DAS-rAG is more ambiguous and cheap-looking compared to the baseline method (DAS). These features are also reflected in the aesthetic score, resulting in a higher score for DAS-AG and a lower score for DAS-rAG.

### Example 2: "Japanese Geisha Girl"

This example also illustrates a successful case of the proposed method (DAS-AG and DAS-rAG).
In the result of the baseline method, the facial features are misaligned, with numerous artifacts.
However, in the result of DAS-AG, the facial features are arranged more accurately, though not perfectly. Additionally, the large water droplet in the lower left appears more photorealistic.

Another notable difference is that DAS-AG chooses an open eye instead of a closed one.
This may suggest that the aesthetic model reflects human preferences for photogenic images, influenced by the human-voted labels in the training dataset.

In contrast, the result of DAS-rAG produces an almost unrecognizable face, with fewer visually interesting elements in the image.

### Example 3: "Gamma-ray Bursts Over Tokyo Tower"

This example highlights a failure case of DAS-AG, where its aesthetic score (3.62) is lower than that of DAS (3.80).

Looking at the generated image, though the cloud details become more fine-grained in DAS-AG, the central tower, which stands out in the baseline method (DAS), disappears (or is pushed far into the background), which may have affected the aesthetic score of the evaluation model.

This could be due to either overfitting to the aesthetic prediction model used during generation,
or a difference in preference between the evaluation model and the generation model, given the inherent ambiguity of "aesthetics".

The result of DAS-rAG successfully deteriorates the aesthetic quality of the generated image, where the texture details in the front tower are diminished, and the cloud formations become more vague, making them almost unrecognizable.

The aesthetic score also reflects this visual degradation, resulting in the lower score of 3.00 compared to DAS's 3.80.

## Appendix

### A. Experimental Setup

**Model & Optimization Parameters**
Parameter | Value
-|-
resolution | 448
#steps | 200
batch size | 8
lr | 0.05
clip weight | 5

**Aesthetic Score Control**
Parameter | Value
-|-
aesthetic range | (0, 3)
aesthetic schedule | exponential decay
aesthetic decay rate | 0.06

**Reproducibility Parameters**

Parameter | Value
-|-
use deterministic algorithm | Yes
interpolation mode | bilinear
seed | 42

**Augmentation Parameters**

Parameter | Value
-|-
texture suppression (TV) | -8
color suppression (L1) | 0.00
gaussian noise range | (0.20, 0.50)
gaussian noise schedule | exponential decay
gaussian noise decay rate | 0.03
color jitter range | (0.05, 0.30)
color jitter schedule | exponential decay
color jitter decay rate | 0.03
max shift | 32

for more details, see `src/app.py`.

### B. Prompt Format

#### B-1. Positive Prompts

##### a. "Gigantic Mona Lisa Attack the City"

```
A photorealistic illustration of "a gigantic metallic Mona Lisa in red pajama attacking the city", dynamic camera angle, fine-grained details, well-recognizable, close-up, well-recognizable human face
```

##### b. "Gamma-ray Burst on Tokyo Tower"
```
A photorealistic illustration of "a massive gamma-ray burst engulfs the towering Tokyo Tower on Earth's final day", dynamic camera angle, fine-grained details, well-recognizable, close-up
```

##### c. "Mt. Fuji, a Hawk, and an Eggplant"

```
A photorealistic illustration of "a stunning Mount Fuji, a majestic hawk, and a symbolic eggplant, set against the neon-lit skyline of cyberpunk Tokyo", dynamic camera angle, fine-grained details, well-recognizable, close-up
```

##### d. "Japanese Geisha Girl"

```
A photorealistic illustration of "a cute Japanese Geisha girl wearing a rainbow-colored kimono, crying with huge drops of tears", fine-grained details, well-recognizable, close-up, well-recognizable face expression
```

#### B-2. Negative Prompts

##### e. Suppress Row-Quality Result

```
text present, low-quality, low-resolution, insane, ugly, grotesque, horrifying, blurred, noisy, distorted, pixelated, artifact present, without depth, flat, boring, uninteresting, unattractive, unappealing, unclear, dull, dark, gloomy, depressing, monotonous
```