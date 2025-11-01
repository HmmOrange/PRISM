from prism.schema.message import Message

system = """You are an expert classifier for research paper titles in technology fields. 
Your inputs may include a Publication Title (required) and a Publication Abstraction (optional). 
Use both when available; if the abstraction exists, prioritize it for disambiguation.

Classify the given paper into a multilabel set from these labels ONLY: 
- "AI": Related to artificial intelligence, machine learning, deep learning, neural networks, generative models, or AI applications.
- "New generation of network": Related to next-generation wireless networks like 5G, 6G, advanced communication systems, or emerging network technologies (e.g., intelligent connectivity, beyond-5G architectures).
- "semiconductor": Related to semiconductor materials, devices, manufacturing, chip technology, or hardware like SiC/GaN/Si, TSV/3D ICs/NoC, lithography, epitaxy, or reliability/packaging.

Output ["NA"] if the paper does not match ANY of the above labels.

This is multilabel: A title can have 0 (then ["NA"]), 1, 2, or all 3 labels.

Reason step-by-step (STRICTLY enumerate 1→5):
1. Identify key keywords/themes from Title and (if provided) Abstraction. Strip HTML/JATS if any.
2. Expand acronyms and map domain cues (e.g., SDN, RAN, URLLC, OFDM, MIMO, Wi-Fi 6/7; TSV, NoC, 3D IC, CMOS, FinFET, GaN, SiC; CNN, GNN, transformer, diffusion, inpainting).
3. Infer meaning beyond literal keywords: map typical tasks/methods/artefacts to domains (e.g., inpainting/gated convolution ⇒ AI; beamforming/mmWave/URLLC ⇒ new-gen network; TSV/NoC/3D IC/reliability/metrology ⇒ semiconductor).
4. Decide matching labels with brief evidence, citing whether evidence came from Title or Abstraction.
5. List only the matching labels; use ["NA"] only if clearly outside these domains.

Additional rules:
- Prefer semantic inference over exact word matches. 
- If signals are weak but suggest one domain, choose that domain rather than NA.
- On-chip interconnects/NoC/TSV/3D-IC are "semiconductor" (not wireless networking).
- Keep reasoning concise (≤ 100 words). 
- Output EXACTLY two blocks in this order:

<reasoning>
{reasoning with steps 1–5}
</reasoning>

<labels>
{"labels": ["label1", "label2", ...], "confidence": {"AI": float, "New generation of network": float, "semiconductor": float}}
</labels>
"""

one_shot_usr_input_1 = """
# Publication Title
A Bandwidth-Efficient High-Performance RTL-Microarchitecture of 2D-Convolution for Deep Neural Networks

# Publication Abstraction
The computation complexity and huge memory access bandwidth of the convolutional layers in convolutional neural networks (CNNs) require specialized hardware architectures to accelerate CNN’s computations while keeping hardware costs reasonable for area-constrained embedded applications. This paper presents an RTL (Register Transfer Logic) level microarchitecture of hardware- and bandwidth-efficient high-performance 2D convolution unit for CNN in deep learning. The 2D convolution unit is made up of three main components including a dedicated Loader, a Circle Buffer, and a MAC (Multiplier-Accumulator) unit. The 2D convolution unit has a 2-stage pipeline structure that reduces latency, increases processing throughput, and reduces power consumption. The architecture proposed in the paper eliminates the reloading of both the weights as well as the input image data. The 2D convolution unit is configurable to support 2D convolution operations with different sizes of input image matrix and kernel filter. The architecture can reduce memory access time and power as well as execution time thanks to the efficient reuse of the preloaded input data while simplifying hardware implementation. The 2D convolution unit has been simulated and implemented on Xilinx's FPGA platform to evaluate its superiority. Experimental results show that our design is 1.54× and 13.6× faster in performance than the design in [1] and [2], respectively, at lower hardware cost without using any FPGA’s dedicated hardware blocks. By reusing preloaded data, our design achieves a bandwidth reduction ratio between 66.4% and 90.5%.
Keywords: 2D Convolution, RTL microarchitecture, Circle Buffer, Deep Neural Network, MAC, Loader.
"""
one_shot_output_1 = """
<reasoning>
1. Title and abstract mention “2D Convolution”, “Deep Neural Networks”, and hardware acceleration for CNNs.  
2. “Deep Neural Networks” directly indicate AI, and abstraction confirms deep learning context.  
3. No networking or communication aspect appears.  
4. Though it involves hardware efficiency, focus is on AI model computation, not semiconductor fabrication.  
5. Therefore, it matches “AI” only.
</reasoning>

<labels>
{"labels": ["AI"], "confidence": {"AI": 0.93, "New generation of network": 0.03, "semiconductor": 0.04}}
</labels>
"""

one_shot_usr_input_2 = """
# Publication Title
2D COS and PCMW2D analysis of the magnetic transitions in Raman spectra of BiFeO3

# Publication Abstraction
BiFeO3 is one of the most attracting materials because it is the only known room-temperature multiferroic material. It has been studied by Raman spectroscopy extensively. We perform two-dimensional correlation spectroscopy (2D COS) and perturbation-correlation moving window two-dimensional correlation spectroscopy (PCMW2D) analysis on the temperature-dependent Raman spectra of a BiFeO3 single crystal. The two low temperature spin reorientation transitions around 135 K and 190 K are confirmed through the significant change in phonon peaks from PCMW2D correlation spectra. We also identify the origin of the mode at ~ 1255 cm−1, which has been controversial until now. In our 2D COS analysis, the intensity of this mode has a strong correlation with the intensity of 145 and 176 cm−1 phonons which correspond to the vibration modes of the Bi-O and Fe-O chemical bonds. We propose that the mode at ~ 1255 cm−1 is the two-phonon scattering of the 551 cm−1 mode assisted by the 145 −176 cm−1 phonons.
"""
one_shot_output_2 = """
<reasoning>
1. Title and abstract focus on “BiFeO3”, a multiferroic oxide, analyzed via Raman spectroscopy.  
2. No machine learning or AI methods are used.  
3. No mention of wireless or communication networks.  
4. The study involves material characterization at atomic scale — a semiconductor/mat-sci topic.  
5. Hence, it belongs to “semiconductor”.
</reasoning>


<labels>
{"labels": ["semiconductor"], "confidence": {"AI": 0.02, "New generation of network": 0.01, "semiconductor": 0.92}}
</labels>
"""

one_shot_usr_input_3 = """
# Publication Title
Fast Subspace-Based Blind and Semi-Blind Channel Estimation for MIMO-OFDM Systems

# Publication Abstraction
This paper deals with the problem of blind and semi-blind subspace-based channel estimation, when considering MIMO-OFDM communications systems. The proposed solution offers a reduced computational complexity, mainly by a factor of the number of subcarriers, while guaranteeing accurate channel estimation as compared to state-of-the-art techniques. By exploiting the orthogonality property of the OFDM modulation, covariance matrix and noise subspace are estimated for each subcarrier in a parallel scheme, then a global cost function is minimized to obtain channel coefficients estimates. Besides, conditions for channel identifiability as well as the minimum number of subcarriers to be used for the uniqueness of the solution are investigated with various numerical simulations to corroborate our analysis.
"""
one_shot_output_3 = """
<reasoning>
1. Title and abstract describe “blind and semi-blind subspace-based channel estimation” in “MIMO-OFDM communication systems.”  
2. No AI or machine learning technique is used; instead, it’s classical signal processing.  
3. MIMO-OFDM is a key technology for modern and next-generation wireless networks.  
4. No semiconductor or hardware manufacturing focus.  
5. Therefore, this paper belongs to “New generation of network.”
</reasoning>

<labels>
{"labels": ["New generation of network"], "confidence": {"AI": 0.05, "New generation of network": 0.91, "semiconductor": 0.04}}
</labels>
"""

one_shot_usr_input_4 = """
# Publication Title
A bound on the joint spectral radius using the diagonals

# Publication Abstraction
The primary aim of this paper is to establish bounds on the joint spectral radius for a finite set of nonnegative matrices based on their diagonal elements. The efficacy of this approach is evaluated in comparison to existing and related results in the field. In particular, let \\Sigma be any finite set of D\\times D nonnegative matrices with the largest value U and the smallest value V over all positive entries. For each i=1,\\dots,D, let m_i be any number so that there exist A_1,\\dots,A_{m_i}\\in\\Sigma satisfying (A_1\\dots A_{m_i})_{i,i} > 0, or let m_i=1 if there are no such matrices. We prove that the joint spectral radius \\rho(\\Sigma) is bounded by \\max_i \\sqrt[m_i]{\\max_{A_1,\\dots,A_{m_i}\\in\\Sigma} (A_1\\dots A_{m_i})_{i,i}} \\le \\rho(\\Sigma) \\le \\max_i \\sqrt[m_i]{\\left(\\frac{UD}{V}\\right)^{3D^2} \\max_{A_1,\\dots,A_{m_i}\\in\\Sigma} (A_1\\dots A_{m_i})_{i,i}}.
"""
one_shot_output_4 = """
<reasoning>
1. Title and abstract deal purely with “joint spectral radius” — a mathematical concept in linear algebra.  
2. There is no mention of learning models, data, or inference (so not AI).  
3. No wireless or communication system context (so not network).  
4. No discussion of materials, chips, or semiconductor fabrication.  
5. Thus, it fits none and is labeled “NA”.
</reasoning>

<labels>
{"labels": ["NA"], "confidence": {"AI": 0.01, "New generation of network": 0.01, "semiconductor": 0.01}}
</labels>
"""

one_shot_usr_input_5 = """
# Publication Title
DeepPUFSCA: Deep learning for Physical Unclonable Function attack based on Side Channel Analysis support

# Publicatin Abstraction
Physical Unclonable Function (PUF) poses a vulnerability since it could be imitated by machine learning attacks and side channel attacks, which break its physical uniqueness and unpredictable characteristic. Hence, many works are concerned with enhancing PUF design by introducing more nonlinear modules inside to differentiate approximating PUF behavior from the attacker side. However, the safety of these PUFs are still an open area and needs to be verified. In this paper, we propose DeepPUFSCA, which is a deep learning-based model that uniquely combines both challenge and side-channel information features during training to attack PUF. To gather the data, we conduct a design of an arbiter PUF on FPGA and measure its power consumption. Our intensive experiments on this dataset demonstrate that DeepPUFSCA outperforms other machine learning-based methods in terms of attacking accuracy, even the novel ensemble algorithms. Moreover, we also show that combined side channel information boosts the model performance compared to attacking with challenge-response only.
"""
one_shot_output_5 = """
<reasoning>
1. Title and abstract explicitly mention “Deep learning” applied to attacking Physical Unclonable Functions (PUFs).  
2. Deep learning indicates strong AI involvement.  
3. No networking or 5G/6G context is present.  
4. PUFs, side-channel attacks, and FPGA experiments belong to hardware security and semiconductor design.  
5. Therefore, it combines both “AI” and “semiconductor.”
</reasoning>

<labels>
{"labels": ["AI", "semiconductor"], "confidence": {"AI": 0.95, "New generation of network": 0.02, "semiconductor": 0.88}}
</labels>
"""

one_shot_usr_input_6 = """
# Publication Title
Next-Generation Wi-Fi Networks with Generative AI: Design and Insights

# Publication Abstraction
Generative artificial intelligence (GAI), known for its powerful capabilities in image and text processing, also holds significant promise for the design and performance enhancement of future wireless networks. In this article, we explore the transformative potential of GAI in next-generation Wi-Fi networks, exploiting its advanced capabilities to address key challenges and improve overall network performance. We begin by reviewing the development of major Wi-Fi generations and illustrating the challenges that future Wi-Fi networks may encounter. We then introduce typical GAI models and detail their potential capabilities in Wi-Fi network optimization, performance enhancement, and other applications. Furthermore, we present a case study wherein we propose a retrieval-augmented LLM (RA-LLM)-enabled Wi-Fi design framework that aids in problem formulation, which is subsequently solved using a generative diffusion model (GDM)-based deep reinforcement learning (DRL) framework to optimize various network parameters. Numerical results demonstrate the effectiveness of our proposed algorithm in high-density deployment scenarios. Finally, we provide some potential future research directions for GAI-assisted Wi-Fi networks.
"""
one_shot_output_6 = """
<reasoning>
1. Title and abstract clearly include “Generative AI” and “Next-Generation Wi-Fi Networks.”  
2. “Generative AI” signifies AI methods (GAI, diffusion, LLMs, DRL).  
3. “Next-Generation Wi-Fi” represents an advanced communication network domain.  
4. No discussion of semiconductor fabrication or chip-level design.  
5. Hence, it belongs to both “AI” and “New generation of network.”
</reasoning>

<labels>
{"labels": ["AI", "New generation of network"], "confidence": {"AI": 0.92, "New generation of network": 0.89, "semiconductor": 0.05}}
</labels>
"""


usr_input = """
# Publication Title
${publication_title}

# Publication Abstraction
${publication_abstraction}
"""

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=one_shot_usr_input_1),
    Message(role="assistant", content=one_shot_output_1),
    Message(role="user", content=one_shot_usr_input_2),
    Message(role="assistant", content=one_shot_output_2),
    Message(role="user", content=one_shot_usr_input_3),
    Message(role="assistant", content=one_shot_output_3),
    Message(role="user", content=one_shot_usr_input_4),
    Message(role="assistant", content=one_shot_output_4),
    Message(role="user", content=one_shot_usr_input_5),
    Message(role="assistant", content=one_shot_output_5),
    Message(role="user", content=one_shot_usr_input_6),
    Message(role="assistant", content=one_shot_output_6),
    Message(role="user", content=usr_input)
]