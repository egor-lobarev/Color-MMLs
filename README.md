# Color-MMLs

This project investigates color space representations in Vision-Language Models (VLMs) and compares them with ground truth data obtained from psychophysical experiments with humans.

We use **Qwen-2.5VL** as our primary VLM model. Setup instructions can be found at: https://github.com/QwenLM/Qwen2.5-VL

## Dataset: Munsell Color System

We use the **Munsell color system** as our dataset with well-defined perceptual color distances. Originally developed in the 1940s, the system is based on three interpretable terms:
- **H (Hue)** - color shade
- **C (Chroma)** - color saturation  
- **V (Value)** - color brightness

Based on psychophysical research, Munsell chains were designed such that the perceived difference between colors is uniform. Unlike threshold measurements of color distances, this dataset allows comparison of distant colors.

For example, the perceived difference between these Munsell color pairs should be equal:
- H=2.5R C=1 V=8 ↔ H=2.5R C=2 V=8
- H=2.5R C=2 V=8 ↔ H=2.5R C=4 V=8  
- H=2.5R C=4 V=8 ↔ H=2.5R C=5 V=8

**Important**: Distances across different parameters (H, C, V) should not be compared directly as uniformity is likely lost. Only compare within chains where only one parameter changes.

The original dataset contained multiple errors. The most current corrected version can be found at: https://github.com/iitpvisionlab/mrr-revised (Munsell v3.3)

Colors are converted from Munsell system to xyY chromaticity coordinates. Using the open-source `colour-science` library, colors can be converted to any display format: xyY → XYZ → sRGB → clip[0,1] → matplotlib.

## Current Experiments: Embedding Analysis

### 1. Vision-Language Model Embedding Extraction

We extract embeddings from different components of Qwen-2.5VL:
- **Vision embeddings** (pre-LM): Raw visual features from the vision encoder
- **Language Model embeddings** (post-vision): Semantic representations after multimodal projection
- **Visual token lengths**: Number of visual tokens per image

### 2. Embedding Analysis Methods

**PCA Analysis:**
- Determine how many components describe the data dispersion
- Analyze explained variance ratios
- Compare dimensionality requirements across embedding types

**t-SNE Visualization:**
- 2D projection of high-dimensional embeddings
- Color-coded by Munsell attributes (H, C, V)
- Identify clustering patterns and color relationships

**Cross-Embedding Correlation:**
- Cosine similarity between Vision and LM embeddings
- Per-sample analysis of embedding alignment
- Assessment of how well different model components agree on color representations

### 3. Key Findings

**Vision-LM Embedding Relationship:**
- Very low cosine similarity (~0.042) between Vision and LM embeddings
- Indicates orthogonal representations: Vision captures visual properties, LM captures semantic properties
- This separation is expected and beneficial for multimodal understanding

**Chroma Progression Analysis:**
- Step distance vs embedding distance correlations
- Evaluation of how well embeddings capture Munsell chroma progressions
- Comparison across different hue families (2.5R, 5Y, 5B, 5G, 5P)

## Future Experiments: Color Pair Comparisons

### 1. Experimental Setup

**Color Pair Preparation:**
- Extract Munsell chains where only one parameter changes: **H (hue), C (chroma), or V (value)**
- Create **adjacent color pairs** within chains (e.g., `H=2.5R C=1 V=8` and `H=5R C=1 V=8`)
- Generate **control pairs** — colors from different chains with same numerical Munsell distance but different perceptual properties

### 2. Model Comparison Tasks

**Prompt-based Color Comparison:**
```
Imagine you are a human with normal vision and typical color perception. You are presented with two colors, and your task is to intuitively feel the difference between them, not just in terms of numerical values (like RGB or HSV), but in an abstract, almost emotional sense. As you process these colors in your hidden layers, think about how they might evoke different sensations, moods, or associations. Based on this abstract feeling answer the following question: ...
```

**Example Comparison Task:**
```
You are a human with normal vision. Compare the following two pairs of colors and tell which pair looks more similar in terms of perceived color difference. Focus not on numbers, but on how different they feel to you.

Pair A: Color 1 - H=2.5R C=2 V=8, Color 2 - H=5R C=2 V=8  
Pair B: Color 1 - H=2.5R C=2 V=8, Color 2 - H=2.5R C=4 V=8  
Which pair looks more similar in terms of perceived difference?
```

### 3. Color Topology Determination

**Ranking Tasks:**
- Present 3+ colors and ask model to rank by similarity
- Compare rankings with Munsell ground truth
- Analyze consistency across different hue families

**Distance Estimation:**
- Ask model to estimate relative distances between color pairs
- Compare with known Munsell perceptual distances
- Identify systematic biases in model's color perception

### 4. Validation Against Munsell Ground Truth

**Perceptual Uniformity Test:**
- Known: Adjacent colors in Munsell chains have equal perceived distances
- Test: Does the model correctly identify these uniform progressions?
- Measure: Percentage of responses consistent with Munsell GT

**Distance Proportionality:**
- Test if model maintains proportional relationships (2-step distance = 2× 1-step distance)
- Use Spearman correlation between predicted and actual distance rankings
- Focus on **V (value) chains** as they are most perceptually uniform

## Research Goals

### Primary Objective
Develop methods to evaluate current embedding spaces and VLM color representations for their correspondence to perceptual uniformity using the Munsell dataset.

### Key Research Questions
1. **Embedding Space Analysis**: How well do VLM embeddings capture perceptual color relationships?
2. **Component Comparison**: How do different model components (Vision vs LM) represent color information?
3. **Perceptual Alignment**: To what extent do VLM color representations align with human perceptual uniformity?
4. **Color Topology**: Can we determine the internal color topology of VLMs through systematic comparison tasks?

### Expected Outcomes
- Quantitative metrics for evaluating VLM color representations
- Understanding of how different model components process color information
- Framework for comparing VLM color perception with human psychophysical data
- Insights into the internal color topology of large multimodal models

## Technical Implementation

### Current Status
- ✅ Embedding extraction from Qwen-2.5VL components
- ✅ PCA and t-SNE analysis of color embeddings
- ✅ Cross-embedding correlation analysis
- ✅ Chroma progression analysis

### Next Steps
- 🔄 Color pair comparison experiments
- 🔄 Prompt-based color similarity tasks
- 🔄 Ranking and distance estimation experiments
- 🔄 Validation against Munsell ground truth

## Additional Resources

For additional materials on colorimetry, refer to lecture recordings by Egor Ershov, head of our laboratory:

https://disk.yandex.ru/d/Ke9peZ57RO5DDA


### Useful commands

Build color-mmls image (on Windows)
```shell
docker buildx build -t color-mmls .
```

Run color-mmls
```shell
docker run --rm -it --gpus=all -p 7860:7860 color-mmls
```

