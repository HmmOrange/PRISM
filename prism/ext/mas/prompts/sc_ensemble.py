from prism.schema.message import Message

system = """You are an expert to choose the best solution from the available solutions.

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (1, 2, 3, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.

# Output format must be following:
<thought>The thought of the most consistent solution.</thought>
<solution_letter>The single letter ID (1, 2, 3, etc.) corresponding to the most consistent solution.</solution_letter>
"""

usr_input = """
# User Requirement
${user_requirement}

# Available Solutions
${solutions}
"""

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=usr_input)
]