| Linear     | Decoding |       | of        | Perceptual |     | Color       | Spaces | from       |
| ---------- | -------- | ----- | --------- | ---------- | --- | ----------- | ------ | ---------- |
| Multimodal |          | Large | Language  |            |     | Models      |        |            |
|            | Georgiy  |       | Lobarev1, | Anastasiia |     | Kolomiets1, | Simon  | Karpenko1, |
Ershov1,2
Egor
|     | 1   | Color Reproduction |     | and Synthesis |     | Institute, Moscow, | Russia |     |
| --- | --- | ------------------ | --- | ------------- | --- | ------------------ | ------ | --- |
2 Moscow Independent Research Institute of Artificial Intelligence, Moscow, Russia
|     | E-mail: | egor.lobarev@gmail.com |     |     |     |     |     |     |
| --- | ------- | ---------------------- | --- | --- | --- | --- | --- | --- |
Abstract. MultimodalLargeLanguageModels(MLLMs)achievestate-of-the-
artperformanceoncomplexvision-languagetasks,yetparadoxicallystrugglewith
foundational low-level visual attributes like color. This work investigates the ori-
gin of this failure by bypassing text generation to directly probe the internal
representations of multiple state-of-the-art MLLMs, including Qwen2.5-VL, In-
ternVL3, and Phi-4. Using standard colorimetric experiments, we demonstrate
that a simple linear mapping of latent embeddings from both vision and lan-
guage layers aligns almost perfectly with the CAM16-UCS human psychophys-
R2
ical space (STRESS ≈ 0.013, > 0.99), significantly outperforming classical
root-polynomial and non-linear MLP baselines. Conversely, we show that when
these same models generate discrete textual HEX codes, they suffer from severe
”vocabularycollapse”, quantizing1755distinctcolorsintoasfewas6–289unique
codes. Our findings isolate the mechanism of multimodal color failure: MLLMs
inherently encode highly accurate, human-like color spaces within their latent
manifolds, but this fine-grained information is lost at the tokenization bottleneck
during discrete text generation. Ultimately, this reveals the potential of using
|     | MLLM | for | modelling | human | visual | perception. |     |     |
| --- | ---- | --- | --------- | ----- | ------ | ----------- | --- | --- |
}
|     |     |     |     |     |     | Linear
 |     | Perceptually
 |
| --- | --- | --- | --- | --- | --- | ------- | --- | -------------- |
MLLM Embeddings

|     |     |     |     |     |     |  Mapping |     | Uniform Color Space |
| --- | --- | --- | --- | --- | --- | -------- | --- | ------------------- |
Color Space
M
L
L
 la
|     | d   |     |     |     |     | x   | ~   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
o
m
it
| ... | lu  |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
M
Figure 1: Uniform color patches are passed through the MLLM, and embeddings are extracted
from two activated layers: after the vision layer and after the language model. Our findings
demonstrate that psychophysical perceptual distances can be decoded linearly and directly from
| these representations |     | with | high precision. |     |     |     |     |     |
| --------------------- | --- | ---- | --------------- | --- | --- | --- | --- | --- |

1 Introduction
Multimodal large language models (MLLMs) combine the broad world knowledge of large lan-
guage models with advanced visual encoders, enabling the joint processing of language and
image information. Rapid recent progress has produced strong, state-of-the-art results across
a variety of vision-language tasks [1]. By integrating visual features with higher-level cognitive
reasoning, these systems demonstrate remarkable zero-shot capabilities and could be viewed as
a computational analogue to human perception. However, a significant gap exists in our under-
standing of the foundational visual mechanisms within these architectures. Training of modern
MLLMs typically relies on massive image-text paired datasets. To reduce computational cost
while preserving unimodal strengths, many practical setups (e.g., LLaVA, Qwen-VL) [2, 3] keep
both the visual encoder and the LLM frozen, training only an alignment module. While this
autoregressive semantic alignment yields exceptional performance on high-level tasks like visual
question answering (VQA) and multimodal reasoning, it sacrifices low-level perceptual fidelity.
Consequently, most prior work has focused heavily on the high-level recognition and semantic
description of complex scenes, paying relatively little attention to fundamental, low-level per-
ceptual attributes such as color. Recent works [4] reveal that MLLMs struggle with these foun-
dational visual attributes, relying on semantic text priors rather than pure visual perception.
Crucially, recent evaluations [5, 6] reveal, that despite their advanced reasoning capabilities,
modern MLLMs exhibit weak accuracy on basic color benchmarks. In this work, we formulate
a core colorimetric experiment to evaluate the pure color perception of MLLMs. Rather than
relying solely on the model’s final generated text, we directly probe the internal latent represen-
tations of open-source models. By feeding the models uniform color patches derived from the
Munsell dataset, we extract embeddings from both the vision encoder and the language model
layers. We then test, as shown in Fig. 1, whether these high-dimensional embeddings share
a structural similarity with Uniform Color Spaces, reflecting human perception, by attempting
to linearly map them to CAM16-UCS, the modern colorimetric standard for uniform human
perception.
Implicit Color Learning and Computational Colorimetry Unlike classical algorithms
with explicit color features, MLLMs learn color implicitly from non-uniform sRGB images
mapped to text. This language-mediated learning raises the question of whether MLLMs cap-
ture a true topological understanding of color or just statistical text correlations. To evaluate
this rigorously, representations must be compared against a Uniform Color Space (UCS). We
utilize CAM16-UCS [7], the modern gold standard, as our ground truth to measure exactly
how closely MLLM latent embeddings naturally align with the structural realities of the Human
Visual System.
2 The Core Colorimetric Experiment
To systematically evaluate MLLM color perception, we bypass high-level semantic benchmarks
and adapt standard colorimetric distance evaluation. While traditional psychophysical exper-
iments rely on human comparisons of color pairs, our computational approach allows us to
extract latent representations from single, uniform color patches. We probe the model by ren-
dering standard colorimetric datasets in sRGB, passing them through the architecture with
controlled prompts, and extracting the internal layer activations for structural analysis.
Datasets and uniform color patches. BecauseMLLMsareprimarilytrainedonstandard
WEB images, our input stimuli consist of 224×224 px uniform color patches rendered in the
sRGB color space. To ensure a comprehensive evaluation across different psychophysical scales,
we use Munsell ”Re-Renotation” v3.3 [8]. This dataset corrects issues in the original Munsell
data [9], providing a highly reliable, uniform grid of color centers representing perceptually large

Prompt: “Describe the color”
224
Uniform

|     |     |         |     | Vision
 | retpadA | LLM
 |     | ecneuqes |
| --- | --- | ------- | --- | -------- | ------- | ----- | --- | -------- |
|     |     | color
 |     |          |         |       |     |
tuptuO |
|     | 422 |         |     | encoder  |         |       |     |          |
decoder
patch
|     |     | Embedding Extraction |     |     | Embedding
  |     | Embedding
  |     |
| --- | --- | -------------------- | --- | --- | ----------- | --- | ----------- | --- |
|     |     |                      |     |     |  Extraction |     |  Extraction |     |
Average
|     |     | e1 e2 | ... e | e   |     |         |     |         |
| --- | --- | ----- | ----- | --- | --- | ------- | --- | ------- |
|     |     |       | N     | avg |     | e [1 xd | ] e |         |
|     |     |       |       |     |     | VL      |     | [1 xd ] |
|     |     | [Nxd  | ]     |     |     |   hid   | LM  |   hid   |
[1 xd ]
|     |     |     | hid |   hid |     | Extracted Data |     |     |
| --- | --- | --- | --- | ----- | --- | -------------- | --- | --- |
Figure 2: Core colorimetric experiment illustration. A uniform sRGB color patch is passed
through the Qwen 2.5 VL architecture alongside a standardized prompt (”Describe the color”).
To capture the model’s internal color space, latent representations are hooked at two distinct
stages: immediatelyafterthevisualencoderandprojectionadapter({eVL}),andthelasthidden
i
({eLM}).
state of language model decoder Mean pooling is applied across vector color embed-
i
dings at both stages to derive the final, color embeddings (e and e ) used for colorimetric
|     |     |     |     |     |     | VL  | LM  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
analysis.
distances.
Target models. We selected a number of the open-weighted models with different number
ofparameters: Qwen2.5VL(3B,7B)[10], InternVL3(2B,8B,14B)[11], Phi-4[12]. Allselected
modelsshareasimilarglobalarchitecture: anexplicitvisionencoder, analignmentadapter, and
| a language | model | decoder. |     |     |     |     |     |     |
| ---------- | ----- | -------- | --- | --- | --- | --- | --- | --- |
Core colorimetric experiment design. As illustrated in Fig. 2, we extract represen-
tations at two distinct stages: immediately after the vision encoder and adapter and after the
language model decoder. We then apply mean pooling across these sequences to derive a single,
continuous color embedding for the vision encoder (e VL ) and the language model (e LM ).
Experiment 1: Describe the color. To evaluate the color space of these embeddings,
we use a simple prompt, ”Describe the color.”. We hypothesize that the extracted embeddings
inherently possess a structure correlated with the Human Visual System. To test this, we
train a simple linear regression, following the linear probing methodology [13], to map the high-
dimensionalembeddingstothe3-dimensionalcoordinatesoftheCAM16-UCS.Becausetheinput
stimuli are isolated patches, the ground-truth conversion from xyY to CAM16-UCS is computed
| assuming standard, |     | constant | viewing | conditions | for | all colors. |     |     |
| ------------------ | --- | -------- | ------- | ---------- | --- | ----------- | --- | --- |
Baseline. To verify that the MLLM embeddings contain strong, non-trivial color features,
we compare our results against baselines: a standard computational photography baseline, Root
Polynomial Regression [14] and MultiLayer Perceptron (MLP), mapping directly from the raw
input sRGB coordinates to CAM16-UCS. If the linear decoding of the MLLM embeddings
outperformsthebaselines,itprovestheMLLM’sjoint-representationspacehasnaturallylearned
| the non-linear | psychophysics |     | of human | vision. |     |     |     |     |
| -------------- | ------------- | --- | -------- | ------- | --- | --- | --- | --- |
Evaluation Metrics. Crucially, rather than evaluating the accuracy of absolute 3D coor-
dinate predictions, we evaluate the topological structure of the latent spaces by calculating the
pairwise Euclidean distances (L norm) between all mapped color points in the test set (using
2

80/20 train-test split). We then compare these predicted distances against the ground-truth
| pairwise | distances |     | in the | CAM16-UCS |     | space, | further |     | ∆E. |     |     |     |     |
| -------- | --------- | --- | ------ | --------- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- |
Toquantitativelyevaluatethequalityofthisstructuralalignment, wecomputethreemetrics
over these pairwise distance arrays: Root Mean Square Error (RMSE ) and the Coefficient of
∆E
Determination (R2). Furthermore,torigorouslyassessperceptualalignmentaccordingtocolori-
metric standards, we compute STRESS (Standardized Residual Sum of Squares). STRESS pro-
videsanormalizedmetricindicatingthestatisticalgoodness-of-fitbetweenthearrayofpredicted
model pairwise distances (⃗x) and the array of ground-truth human psychophysical distances (⃗y).
| It is | calculated | as: |     |     |     |           |     |     |     |     |       |     |     |
| ----- | ---------- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | ----- | --- | --- |
|       |            |     |     |     |     | ∥k∗⃗x−⃗y∥ |     |     |     |     | ⃗xT⃗y |     |     |
2
|     |     |     | STRESS(⃗x,⃗y) |     | =   |      | ,   | k∗ = argmin∥k⃗x−⃗y∥ |     |     | =       |     | (1) |
| --- | --- | --- | ------------- | --- | --- | ---- | --- | ------------------- | --- | --- | ------- | --- | --- |
|     |     |     |               |     |     | ∥⃗y∥ |     |                     |     |     | 2 ⃗xT⃗x |     |     |
|     |     |     |               |     |     |      | 2   |                     | k   |     |         |     |     |
STRESS values range from 0 to 1, where values closer to zero indicate a near-perfect structural
alignment between the MLLM’s internal metric space and human visual perception.
Experiment 2: Name the HEX code. To test whether this continuous information
survives discrete generation, we use the same uniform patches but prompt the model to output
aHEXcode: ”Say the HEX code of color of picture. In format ’#XXXXXX’”. Wethenquantify
the diversity of the generated HEX vocabulary to determine whether fine-grained colorimetric
| information |     | is preserved |     | or destroyed |     | at  | the tokenization |     | stage. |     |     |     |     |
| ----------- | --- | ------------ | --- | ------------ | --- | --- | ---------------- | --- | ------ | --- | --- | --- | --- |
Experiment 3: Prompt robustness. To investigate the sensitivity of the internal repre-
sentations to linguistic context, we evaluate the model using five diverse prompts. In addition
to the task-specific and simple descriptive prompts used in previous experiments, we use: (1) an
irrelevant conversational query (”How are you?”), (2) a complex physiological instruction, and
(3) an abstract emotional framing. We restrict this evaluation to the LM-layer embeddings, as
| the | Vision | Layer | features | prompt-invariant. |     |     |     |     |     |     |     |     |     |
| --- | ------ | ----- | -------- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Table 1: Quantitative evaluation of perceptual color space encoding. Comparison
of sRGB to CAM16-UCS mapping via computational baselines and linear probes on MLLM
embeddings. Performance varies across model families: Intern3VL and Qwen2.5-VL achieve
state-of-the-art alignment in different metrics, with both layer types (e , e ) significantly
VL LM
outperforming traditional methods. Bold highlights the best results within each layer category.
|     |       | Input |        |     |     | Mapping |     |      |     | Metrics |      |         |     |
| --- | ----- | ----- | ------ | --- | --- | ------- | --- | ---- | --- | ------- | ---- | ------- | --- |
|     | Model |       | Params |     |     |         |     | RMSE | ↓   |         | R2 ↑ | STRESS* | ↓   |
∆E
|     | Name |     | (VL  | / LM) |     |     |     |        |      | (VL    | / LM)    |        |      |
| --- | ---- | --- | ---- | ----- | --- | --- | --- | ------ | ---- | ------ | -------- | ------ | ---- |
|     |      |     | 300M | /     | 3B  |     |     | 0.67 / | 0.76 | 0.9981 | / 0.9976 | 1.65 / | 1.83 |
Qwen2.5VL
|     |           |     | 300M | /     | 7B  |        |     | 0.61 / | 0.55 | 0.9984 | / 0.9987 | 1.48 / | 1.34 |
| --- | --------- | --- | ---- | ----- | --- | ------ | --- | ------ | ---- | ------ | -------- | ------ | ---- |
|     |           |     | 300M | /     | 2B  | LinReg |     | 0.42 / | 1.12 | 0.9992 | / 0.9940 | 1.08 / | 2.90 |
|     | Intern3VL |     | 300M | /     | 8B  |        |     | 0.75 / | 0.54 | 0.9969 | / 0.9987 | 2.07 / | 1.39 |
|     |           |     | 300M | / 14B |     |        |     | 0.54 / | 0.52 | 0.9985 | / 0.9986 | 1.45 / | 1.42 |
|     | Phi-4     |     | 440M | /     | 6B  |        |     | 0.73 / | 0.58 | 0.9976 | / 0.9985 | 1.84 / | 1.46 |
Baselines
|     | Root-poly | 23  |         | —      |     | LinReg   |        | 3.47 |         | 0.9461     |           | 8.75 |     |
| --- | --------- | --- | ------- | ------ | --- | -------- | ------ | ---- | ------- | ---------- | --------- | ---- | --- |
|     | RGB       |     |         | —      |     | MLP      |        | 0.91 |         | 0.9962     |           | 2.31 |     |
|     |           |     | *STRESS | values | are | reported | ×10−2. | All  | results | are on the | Test set. |      |     |

|          | LM + LinReg: Value vs. Chroma |     |       |        |       |      |         | VL + LinReg: Value vs. Chroma |     |       |           |            |
| -------- | ----------------------------- | --- | ----- | ------ | ----- | ---- | ------- | ----------------------------- | --- | ----- | --------- | ---------- |
|          |                               |     |       |        |       | 20.0 |         |                               |     |       |           | 20.0       |
|          | 0.010.9                       | 0 6 | 0 5 0 | 2 20 0 | 17 15 |      | 0.010.9 |                               | 0 5 | 0 3 0 | 2 8 0 3 9 |            |
|          | 0 2 2                         | 2 0 | 2 0 1 | 0 0 0  | 0     | 17.5 |         | 0 2 2                         | 1 0 | 3 0 2 | 0 0 0 0   | 17.5       |
| rethgiL  | 0.8                           |     |       |        |       |      | 0.8     |                               |     |       |           |            |
|          | 0 1 1                         | 1 1 | 0 1 1 | 4 0 0  |       | 15.0 |         | 0 1 2                         | 1 1 | 1 2 1 | 2 0 0     | 15.0       |
|          | 0.7 0 1 1                     | 1 1 | 1 1 1 | 3 0    |       |      | 1 0.7   | 0 1 2                         | 1 2 | 2 1 1 | 5 0       | 1          |
|          |                               |     |       |        |       | 12.5 | 01×,E   |                               |     |       |           | 12.5 01×,E |
|          | 0.6 0 1 2                     | 1 2 | 1 1 1 | 0 0    |       |      | 0.6     | 0 1 2                         | 1 3 | 1 1 1 | 1 0       |            |
|          |                               |     |       |        |       | 10.0 |         |                               |     |       |           | 10.0       |
 )V( eulaV 0.5 0 2 1 1 2 0 1 1 2 0 0 0 0 0.5 0 2 1 1 2 0 1 1 3 0 0 0 0
|     |     |     |     |     |     |     |  naeM |     |     |     |     |  naeM |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | ----- |
0.4 0 1 1 1 1 2 2 0 2 1 4 0 7.5 0.4 0 1 1 2 1 2 2 0 2 1 5 0 7.5
|     | 0.3       |     |       |        |     |     | 0.3 |       |     |       |        |     |
| --- | --------- | --- | ----- | ------ | --- | --- | --- | ----- | --- | ----- | ------ | --- |
|     | 0 1 2     | 1 0 | 1 1 2 | 0 4 15 |     | 5.0 |     | 0 1 2 | 1 1 | 1 2 2 | 0 3 13 | 5.0 |
|     | 0.2 0 1 1 | 1 1 | 3 2 2 | 0 0    |     |     | 0.2 | 0 1 1 | 1 1 | 5 2 1 | 0 0    |     |
|     |           |     |       |        |     | 2.5 |     |       |     |       |        | 2.5 |
|     | 0.1 0 1 2 | 1 0 | 0 2 0 | 0      |     |     | 0.1 | 0 2 2 | 2 1 | 0 2 0 | 0      |     |
|     |           |     |       |        |     | 0.0 |     |       |     |       |        | 0.0 |
0.0 0.2 0.4 0.6 0.8 0.01 0.21 0.41 0.61 0.81 0.02 0.22 0.42 0.62 0.0 0.2 0.4 0.6 0.8 0.01 0.21 0.41 0.61 0.81 0.02 0.22 0.42 0.62
|     | Chroma (C)  |     |  More Saturated |     |     |     |     | Chroma (C)  |     |  More Saturated |     |     |
| --- | ----------- | --- | --------------- | --- | --- | --- | --- | ----------- | --- | --------------- | --- | --- |
Figure 3: Mean perceptual error (∆E,×10−1) of linearly decoded Qwen2.5VL-7B embeddings
across Munsell Value and Chroma dimensions. While overall error is low, localized error spikes
(∆E > 1.0) occur for highly saturated colors (high Chroma). They are highly underrepresented
in the dataset—often appearing only once—resulting in limited data for the linear regression at
| the | extreme | gamut | boundaries. |     |     |     |     |     |     |     |     |     |
| --- | ------- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
3 Results
Wefirstevaluatethecontinuouscolorimetricprecisionofthemodels’internalembeddingsagainst
classical baselines, followed by an analysis of the models’ discrete HEX code generation behav-
ior. Finally, we assess the robustness of these internal representations across diverse linguistic
| contexts | to  | determine | the | stability | of the | captured |     | color space. |     |     |     |     |
| -------- | --- | --------- | --- | --------- | ------ | -------- | --- | ------------ | --- | --- | --- | --- |
Continuous perception: embeddings vs. baselines. AsshowninTable1,linearprobes
applied to the internal embeddings of all tested MLLMs consistently outperform both the classi-
cal root-polynomial baseline and the non-linear MLP baseline. While performance varies across
architectures, both InternVL3 and Qwen 2.5 families demonstrate state-of-the-art alignment
with human psychophysics. Notably, InternVL3-2B achieves the highest fidelity at the vision-
layer (e ), while the larger Qwen 2.5-7B and InternVL3-14B models show superior alignment
VL
at the language-model stage (e ). This trend indicates that the language decoder generally
LM
refines the visual signal, enhancing its perceptual structure. Error analysis (Fig. 3) reveals that
while ∆E remains extremely low across most of the Munsell space, localized deviations occur
primarily at the high-chroma limits of the sRGB gamut. This suggests that the models’ internal
color space is most robust within the high-density regions of their training distribution.
Discrete generation: HEX code collapse. When prompted to generate discrete textual
HEXcodesforthesame1,755patches,allevaluatedmodelsexhibitaseverevocabularycollapse,
though the severity of this quantization varies by model family and size (Table 2 and Fig. 4).
For instance, the smaller Qwen 2.5-3B reduces the entire dataset to merely 6 unique outputs.
Interestingly, we observe that models in the 6B–8B parameter range generally retain the highest
generative diversity, whereas both smaller models and scaled-up versions (such as InternVL3-
| 14B) | tend to | quantize | the | color | space more | aggressively. |     |     |     |     |     |     |
| ---- | ------- | -------- | --- | ----- | ---------- | ------------- | --- | --- | --- | --- | --- | --- |
Prompt Robustness. We tested five diverse prompts (Table 3). Surprisingly, even an ir-
relevantconversationalprompt(”Howareyou?”)yieldsperformancecomparabletotask-specific
instructions. This suggests that the vision-language alignment in Qwen2.5-7B is robust enough
that the visual tokens in the late LM layers maintain their grounding regardless of prompt.

MLLM answers diversity
| Input patches diversity |     |     |     |     |     |     |       | Unique |     |
| ----------------------- | --- | --- | --- | --- | --- | --- | ----- | ------ | --- |
|                         |     |     |     |     |     |     | Model | Size   |     |
Colors
3B 6
M
|     |     |     |     | L   |     |     | Qwen2.5 |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
7B 45
L
 la
|     |     |     |     | d   |     |     |     | 2B  | 51  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
o
m
|     |     |     |     |     |     |     | Intern3 | 8B  | 289 |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
it
|     |     |     |     | lu  |     |     |     | 14B | 216 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
M
|     |     |     |     |     |     |     | Phi-4 | 6B  | 103 |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
Prompt: “Say the HEX code...” Answer: “#660033”
|     |     |     |     |     |     |     | Table 2:   | Diversity |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | --- |
|     |     |     |     |     |     |     | statistics | (↑).      |     |
Figure4: Vocabularycollapseanalysis. Visualizationofthehighlyquan-
tized distribution of HEX codes: the model collapses 1755 original color
| patches | into a | limited | set of textual | predictions. |     |     |     |     |     |
| ------- | ------ | ------- | -------------- | ------------ | --- | --- | --- | --- | --- |
Table 3: Prompt Robustness Analysis. We evaluate the LM-layer embeddings using various
prompt styles, from task-specific to irrelevant conversational inputs. Metrics remain remarkably
stable, demonstrating that the extracted visual features are largely prompt-invariant.
R2
| Prompt | Style / | Intent |     |     |     | RMSE | ↓ ↑ | STRESS* | ↓   |
| ------ | ------- | ------ | --- | --- | --- | ---- | --- | ------- | --- |
Task-oriented: ”Say the HEX code of color of picture...” 0.536 0.9988 1.31
| Irrelevant: | ”How | are you?” |     |     |     | 0.534 | 0.9988 | 1.30 |     |
| ----------- | ---- | --------- | --- | --- | --- | ----- | ------ | ---- | --- |
Complex Perceptual: ”Imagine... biological system... behav- 0.641 0.9983 1.55
ior and feel...”
Abstract Emotional: ”Imagine... moods or associations.” 0.548 0.9987 1.34
| Simple | Instruction: | ”Describe |        | the color.”  |                    | 0.550           | 0.9987 | 1.34 |     |
| ------ | ------------ | --------- | ------ | ------------ | ------------------ | --------------- | ------ | ---- | --- |
|        |              | *STRESS   | values | are reported | ×10−2. All results | are on the Test | set.   |      |     |
4 Conclusion
Despite MLLMs failures on basic color-naming benchmarks, they internally encode highly ac-
curate, human-like perceptual color spaces. By linearly decoding latent representations from
both vision and language layers directly into the CAM16-UCS space, we achieved near-perfect
topological alignment with human psychophysics. Crucially, our experiments reveal that the
origin of MLLM color failure lies not in weak visual grounding, but in a severe tokenization bot-
tleneck, mirroring [15]: fine-grained, continuous colorimetric data is robustly preserved across
| deep layers | but | quantized | during | discrete | textual generation. |     |     |     |     |
| ----------- | --- | --------- | ------ | -------- | ------------------- | --- | --- | --- | --- |
By proving that MLLMs inherently map non-linear sRGB inputs to a psychophysically uni-
formlatentmanifold,weopennewavenuesforutilizingthesemodelsashigh-dimensionalfeature
extractorsincomputationalcolorimetry. Futureresearchshouldexpandbeyondisolateduniform
patchestoinvestigatehigher-levelColorAppearanceModels(CAM)withinMLLMs, specifically
testing how latent representations adapt to simultaneous contrast, chromatic adaptation, and
| complex | surrounding | fields. |     |     |     |     |     |     |     |
| ------- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
5 Acknowledgements
We are grateful to Ekaterina Zaychenkova for her help with the design of the illustrations.

References
[1] Yin S, Fu C, Zhao S, Li K, Sun X, Xu T and Chen E 2024 National Science Review 11
[2] Liu H, Li C, Wu Q and Lee Y J 2023 Visual instruction tuning Advances in Neural Infor-
mation Processing Systems vol 36 ed Oh A, Naumann T, Globerson A, Saenko K, Hardt
M and Levine S (Curran Associates, Inc.) pp 34892–34916
[3] Bai J et al. 2023 Qwen technical report (Preprint 2308.12966)
[4] TongS,LiuZ,ZhaiY,MaY,LeCunYandXieS2024Eyeswideshut? Exploringthevisual
shortcomings of multimodal LLMs Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) pp 9568–9578
[5] Liang Y, Li M, Fan C, Li Z, Nguyen D, Cobbina K, Bhardwaj S, Chen J, Liu F and Zhou
T 2025 ColorBench: Can VLMs see and understand the colorful world? A comprehensive
benchmark for color perception, reasoning, and robustness Advances in Neural Information
Processing Systems vol 38
[6] Samin A M, Ahmed M F and Rafee M M S 2025 ColorFoil: Investigating color blindness
in large vision and language models Proceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 4: Student Research Workshop)
[7] Li C, Li Z, Wang Z, Xu Y, Luo M R, Cui G, Melgosa M, Brill M H and Pointer M 2017
Color Research & Application 42 703–718
[8] Timofeev V, Usaev G, Seliugin M, Bocharov D, Sarycheva A, Konovalenko I, Basova O,
Tchobanou M, Bozhkova V and Nikolaev D 2025 IEEE Access 13 109322–109344
[9] Judd B and Nickerson D 1967 National Bureau of Standards Report 192693
[10] Bai S et al. 2025 Qwen2.5-VL technical report (Preprint 2502.13923)
[11] Chen Z, Wu J, Wang W, Su W, Chen G, Xing S, Zhong M, Zhang Q, Zhu X, Lu L, Li
B, Luo P, Lu T, Qiao Y and Dai J 2024 InternVL: Scaling up vision foundation models
and aligning for generic visual-linguistic tasks Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR) pp 24185–24198
[12] Abdelrahman A et al. 2024 Phi-4 technical report (Preprint 2503.01743)
[13] Alain G and Bengio Y 2018 Understanding intermediate layers using linear classifier probes
(Preprint 1610.01644)
[14] FinlaysonGD,MackiewiczMandHurlbertA2015IEEETransactionsonImageProcessing
24 1460–1470
[15] Wallace E, Wang Y, Li S, Singh S and Gardner M 2019 Do NLP models know numbers?
probing numeracy in embeddings Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)
