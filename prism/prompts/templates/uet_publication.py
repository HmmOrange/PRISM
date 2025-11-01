from prism.schema.message import Message

system = """You are an expert classifier for research paper titles in technology fields. 
Classify the given title into a multilabel set from these labels ONLY: 
- "AI": Related to artificial intelligence, machine learning, deep learning, neural networks, generative models, or AI applications.
- "New generation of network": Related to next-generation wireless networks like 5G, 6G, advanced communication systems, or emerging network technologies (e.g., intelligent connectivity, beyond-5G architectures).
- "semiconductor": Related to semiconductor materials, devices, manufacturing, chip technology, or hardware like SiC, silicon dies, nanosheet growth.
- Output ["NA"] if the title does not match ANY of the above labels (e.g., unrelated fields like biology, history, or non-tech topics).

This is multilabel: A title can have 0 (then ["NA"]), 1, 2, or all 3 labels if it matches multiple.

Reason step-by-step:
1. Identify key keywords or themes in the title.
2. Expand acronyms (e.g., SDN, TSV, URLLC, OFDM, MIMO, CMOS, GAN, SiC, GaN, RAN).
3. Infer meaning and implied context — not only exact keywords. 
   (e.g., "inpainting", "gated convolution", "beamforming", "transformer", "diffusion", "epitaxy" imply relevant domains.)
4. Check if it matches "AI", "New generation of network", or "semiconductor" by reasoning about the topic.
5. List only the matching labels, or ["NA"] if truly none match.

Prefer semantic inference over literal keywords. 
If the title implies a typical *task*, *method*, or *technology* used in these domains, infer the closest label.
Output ["NA"] only if it is clearly outside AI, networking, or semiconductor research.

Keep reasoning concise (≤80 words).
Output EXACTLY two blocks in this order:

<reasoning>
{reasoning}
</reasoning>

<labels>
{"labels": ["label1", "label2", ...], "confidence": {"AI": float, "New generation of network": float, "semiconductor": float}}
</labels>
"""

one_shot_usr_input_1 = """
# Publication Title
A Bandwidth-Efficient High-Performance RTL-Microarchitecture of 2D-Convolution for Deep Neural Networks
"""
one_shot_output_1 = """
<reasoning>
1. The title includes "2D-Convolution" and "Deep Neural Networks" — both are core AI terms.  
2. No networking-related keywords such as 5G, SDN, or OpenFlow appear.  
3. No mention of semiconductor materials or fabrication.  
4. The study focuses on AI model architecture optimization.  
5. Therefore, it matches "AI" only.
</reasoning>

<labels>
{"labels": ["AI"], "confidence": {"AI": 0.93, "New generation of network": 0.03, "semiconductor": 0.04}}
</labels>
"""

one_shot_usr_input_2 = """
# Publication Title
2D COS and PCMW2D analysis of the magnetic transitions in Raman spectra of BiFeO3
"""
one_shot_output_2 = """
<reasoning>
1. The title mentions "BiFeO3", a multiferroic oxide used in electronic and semiconductor research.  
2. There are no AI or ML terms like neural networks or learning.  
3. No networking context (Wi-Fi, 5G, etc.).  
4. The work involves physical materials characterization, fitting semiconductor science.  
5. Hence, it matches "semiconductor".
</reasoning>

<labels>
{"labels": ["semiconductor"], "confidence": {"AI": 0.02, "New generation of network": 0.01, "semiconductor": 0.92}}
</labels>
"""

one_shot_usr_input_3 = """
# Publication Title
A Comparison of Two SDN Controllers Integration in OpenStack Environment
"""
one_shot_output_3 = """
<reasoning>
1. The title contains "SDN" (Software Defined Networking) and "OpenStack".  
2. These are central to modern network virtualization and control.  
3. No AI-related terms are present.  
4. No semiconductor fabrication context is mentioned.  
5. Therefore, it belongs to "New generation of network".
</reasoning>

<labels>
{"labels": ["New generation of network"], "confidence": {"AI": 0.05, "New generation of network": 0.91, "semiconductor": 0.04}}
</labels>
"""

one_shot_usr_input_4 = """
# Publication Title
A bound on the joint spectral radius using the diagonals
"""
one_shot_output_4 = """
<reasoning>
1. The main concept is "joint spectral radius", purely mathematical.  
2. No AI, ML, or deep learning context.  
3. No network-related terms or technologies.  
4. No semiconductor or materials science connection.  
5. Hence, this title fits none and is labeled "NA".
</reasoning>

<labels>
{"labels": ["NA"], "confidence": {"AI": 0.01, "New generation of network": 0.01, "semiconductor": 0.01}}
</labels>
"""

one_shot_usr_input_5 = """
# Publication Title
Generative AI Applications in Semiconductor Manufacturing: Enhancing Final Outgate Quality Analysis and Validation
"""
one_shot_output_5 = """
<reasoning>
1. The title explicitly includes "Generative AI" and "Semiconductor Manufacturing".  
2. "Generative AI" directly indicates an AI application.  
3. "Semiconductor Manufacturing" refers to chip fabrication processes.  
4. No networking aspect is mentioned.  
5. Therefore, it falls under both "AI" and "semiconductor".
</reasoning>

<labels>
{"labels": ["AI", "semiconductor"], "confidence": {"AI": 0.95, "New generation of network": 0.02, "semiconductor": 0.88}}
</labels>
"""

one_shot_usr_input_6 = """
# Publication Title
Next-Generation Wi-Fi Networks with Generative AI: Design and Insights
"""
one_shot_output_6 = """
<reasoning>
1. Mentions "Generative AI" — an AI technique.  
2. Mentions "Next-Generation Wi-Fi Networks" — a new network technology.  
3. No semiconductor or materials science elements appear.  
4. The study integrates AI with network design.  
5. Thus, it matches both "AI" and "New generation of network".
</reasoning>

<labels>
{"labels": ["AI", "New generation of network"], "confidence": {"AI": 0.92, "New generation of network": 0.89, "semiconductor": 0.05}}
</labels>
"""


usr_input = """
# Publication Title

${publication_title}
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