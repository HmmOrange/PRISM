from prism.schema.message import Message

system = """"""

usr_input = """
# Publication Title
${publication_title}
"""

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=usr_input)
]