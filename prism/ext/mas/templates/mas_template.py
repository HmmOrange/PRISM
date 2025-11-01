mas_template = """
from prism.ext.mas.agent.glue_coder import GlueCoder
from prism.ext.mas.agent.ml_coder import MLCoder
from prism.ext.mas.agent.synthesizer import Synthesizer
from prism.schema.message import Message

class MASPipeline:
    def __init__(self, problem: str, data: list):
        self.problem = problem
        self.glue_coder = GlueCoder(problem=self.problem, data=data)
        self.ml_coder = MLCoder(problem=self.problem, data=data)
        self.synthesizer = Synthesize(problem=self.problem, data=data)
        
    async def __call__(self):
        pass
</graph>
"""