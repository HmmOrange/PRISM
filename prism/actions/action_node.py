from argparse import Action
import re
import json
import typing
from typing import Dict, Type, Any, List, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, create_model, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils.constants import USE_CONFIG_TIMEOUT, MARKDOWN_TITLE_PREFIX
from utils.common import OutputParser, general_after_log
from utils.logs import logger
from provider.llm.base_llm import BaseLLM

from prism.utils.santize import sanitize

TAG = "CONTENT"
LANGUAGE_CONSTRAINT = "Language: Please use the same language as Human INPUT."
FORMAT_CONSTRAINT = f"Format: output wrapped inside [{TAG}][/{TAG}] like format example, nothing else."

SIMPLE_TEMPLATE = """
## context
{context}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow instructions of nodes, generate output and make sure it follows the format example.
"""

class FillMode(Enum):
    CODE_FILL = "code_fill"
    XML_FILL = "xml_fill"
    SINGLE_FILL = "single_fill"
    
def dict_to_markdown(d, prefix=MARKDOWN_TITLE_PREFIX, kv_sep="\n", postfix="\n"):
    markdown_str = ""
    for key, value in d.items():
        markdown_str += f"{prefix}{key}{kv_sep}{value}{postfix}"
    return markdown_str
    
class ActionNode:
    """ActionNode is a tree of nodes."""

    schema: str  # raw/json/markdown, default: ""

    # Action Context
    context: str  # all the context, including all necessary info
    llm: BaseLLM  # LLM with aask interface
    children: dict[str, "ActionNode"]

    # Action Input
    key: str  # Product Requirement / File list / Code
    func: typing.Callable  # 与节点相关联的函数或LLM调用
    params: Dict[str, Type]  # 输入参数的字典，键为参数名，值为参数类型
    expected_type: Type  # such as str / int / float etc.
    # context: str  # everything in the history.
    instruction: str  # the instructions should be followed.
    example: Any  # example for In Context-Learning.

    # Action Output
    content: str
    instruct_content: BaseModel

    # For ActionGraph
    prevs: List["ActionNode"]  # previous nodes
    nexts: List["ActionNode"]  # next nodes
    
    def __init__(
        self,
        key: str,
        expected_type: Type,
        instruction: str,
        example: Any,
        content: str = "",
        children: Optional[dict[str, "ActionNode"]] = None,
        schema: str = "",
    ):
        self.key = key
        self.expected_type = expected_type
        self.instruction = instruction
        self.example = example
        self.content = content
        self.children = children if children is not None else {}
        self.schema = schema
        self.prevs = []
        self.nexts = []
    
    def add_prev(self, node: "ActionNode"):
        """增加前置ActionNode"""
        self.prevs.append(node)

    def add_next(self, node: "ActionNode"):
        """增加后置ActionNode"""
        self.nexts.append(node)

    def add_child(self, node: "ActionNode"):
        """增加子ActionNode"""
        self.children[node.key] = node
    
    def _get_children_mapping(self, exclude=None) -> Dict[str, Any]:
        """获得子ActionNode的字典，以key索引，支持多级结构。"""
        exclude = exclude or []

        def _get_mapping(node: "ActionNode") -> Dict[str, Any]:
            mapping = {}
            for key, child in node.children.items():
                if key in exclude:
                    continue
                # 对于嵌套的子节点，递归调用 _get_mapping
                if child.children:
                    mapping[key] = _get_mapping(child)
                else:
                    mapping[key] = (child.expected_type, Field(default=child.example, description=child.instruction))
            return mapping

        return _get_mapping(self)

    def _get_self_mapping(self) -> Dict[str, Tuple[Type, Any]]:
        """get self key: type mapping"""
        return {self.key: (self.expected_type, ...)}
    
    def get_mapping(self, mode="children", exclude=None) -> Dict[str, Tuple[Type, Any]]:
        """get key: type mapping under mode"""
        if mode == "children" or (mode == "auto" and self.children):
            return self._get_children_mapping(exclude=exclude)
        return {} if exclude and self.key in exclude else self._get_self_mapping()
       
    def create_model_class(cls, class_name: str, mapping: Dict[str, Tuple[Type, Any]]):
        """基于pydantic v2的模型动态生成，用来检验结果类型正确性"""

        def check_fields(cls, values):
            all_fields = set(mapping.keys())
            required_fields = set()
            for k, v in mapping.items():
                type_v, field_info = v
                if ActionNode.is_optional_type(type_v):
                    continue
                required_fields.add(k)

            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")

            unrecognized_fields = set(values.keys()) - all_fields
            if unrecognized_fields:
                logger.warning(f"Unrecognized fields: {unrecognized_fields}")
            return values

        validators = {"check_missing_fields_validator": model_validator(mode="before")(check_fields)}

        new_fields = {}
        for field_name, field_value in mapping.items():
            if isinstance(field_value, dict):
                # 对于嵌套结构，递归创建模型类
                nested_class_name = f"{class_name}_{field_name}"
                nested_class = cls.create_model_class(nested_class_name, field_value)
                new_fields[field_name] = (nested_class, ...)
            else:
                new_fields[field_name] = field_value

        new_class = create_model(class_name, __validators__=validators, **new_fields)
        return new_class
       
    def create_class(self, mode: str = "auto", class_name: Optional[str] = None, exclude=None):
        class_name = class_name if class_name else f"{self.key}_AN"
        mapping = self.get_mapping(mode=mode, exclude=exclude)
        return self.create_model_class(class_name, mapping)
        
    def set_recursive(self, name, value):
        setattr(self, name, value)
        for _, i in self.children.items():
            i.set_recursive(name, value)

    def set_llm(self, llm):
        self.set_recursive("llm", llm)
    
    def set_context(self, context):
        self.set_recursive("context", context)
    
    def get_field_names(self):
        """
        Get the field names associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return model_class.model_fields.keys()
    
    def get_field_types(self):
        """
        Get the field types associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return {field_name: field.annotation for field_name, field in model_class.model_fields.items()}
    
    def _create_children_class(self, exclude=None):
        """使用object内有的字段直接生成model_class"""
        class_name = f"{self.key}_AN"
        mapping = self._get_children_mapping(exclude=exclude)
        return self.create_model_class(class_name, mapping)
    
    def to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""
        nodes = self._to_dict(format_func=format_func, mode=mode, exclude=exclude)
        if not isinstance(nodes, dict):
            nodes = {self.key: nodes}
        return nodes
    
    def _to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""

        # 如果没有提供格式化函数，则使用默认的格式化函数
        if format_func is None:
            format_func = lambda node: node.instruction

        # 使用提供的格式化函数来格式化当前节点的值
        formatted_value = format_func(self)

        # 创建当前节点的键值对
        if (mode == "children" or mode == "auto") and self.children:
            node_value = {}
        else:
            node_value = formatted_value

        if mode == "root":
            return {self.key: node_value}

        # 递归处理子节点
        exclude = exclude or []
        for child_key, child_node in self.children.items():
            if child_key in exclude:
                continue
            # 递归调用 to_dict 方法并更新节点字典
            child_dict = child_node._to_dict(format_func, mode, exclude)
            node_value[child_key] = child_dict

        return node_value
    
    def compile_to(self, i: Dict, schema, kv_sep) -> str:
        if schema == "json":
            return json.dumps(i, indent=4, ensure_ascii=False)
        elif schema == "markdown":
            return dict_to_markdown(i, kv_sep=kv_sep)
        else:
            return str(i)

    def tagging(self, text, schema, tag="") -> str:
        if not tag:
            return text
        return f"[{tag}]\n{text}\n[/{tag}]"
    
    def _compile_f(self, schema, mode, tag, format_func, kv_sep, exclude=None) -> str:
        nodes = self.to_dict(format_func=format_func, mode=mode, exclude=exclude)
        text = self.compile_to(nodes, schema, kv_sep)
        return self.tagging(text, schema, tag)
    
    def compile_instruction(self, schema="markdown", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown template with all/root/children nodes"""
        format_func = lambda i: f"{i.expected_type}  # {i.instruction}"
        return self._compile_f(schema, mode, tag, format_func, kv_sep=": ", exclude=exclude)

    def compile_example(self, schema="json", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown examples with all/root/children nodes"""

        # 这里不能使用f-string，因为转译为str后再json.dumps会额外加上引号，无法作为有效的example
        # 错误示例："File list": "['main.py', 'const.py', 'game.py']", 注意这里值不是list，而是str
        format_func = lambda i: i.example
        return self._compile_f(schema, mode, tag, format_func, kv_sep="\n", exclude=exclude)
    
    def compile(self, context, schema="json", mode="children", template=SIMPLE_TEMPLATE, exclude=[]) -> str:
        """
        mode: all/root/children
            mode="children": 编译所有子节点为一个统一模板，包括instruction与example
            mode="all": NotImplemented
            mode="root": NotImplemented
        schmea: raw/json/markdown
            schema="raw": 不编译，context, lang_constaint, instruction
            schema="json"：编译context, example(json), instruction(markdown), constraint, action
            schema="markdown": 编译context, example(markdown), instruction(markdown), constraint, action
        """
        if schema == "raw":
            return f"{context}\n\n## Actions\n{LANGUAGE_CONSTRAINT}\n{self.instruction}"

        ### 直接使用 pydantic BaseModel 生成 instruction 与 example，仅限 JSON
        # child_class = self._create_children_class()
        # node_schema = child_class.model_json_schema()
        # defaults = {
        #     k: str(v)
        #     for k, v in child_class.model_fields.items()
        #     if k not in exclude
        # }
        # instruction = node_schema
        # example = json.dumps(defaults, indent=4)

        # FIXME: json instruction会带来格式问题，如："Project name": "web_2048  # 项目名称使用下划线",
        # compile example暂时不支持markdown
        instruction = self.compile_instruction(schema="markdown", mode=mode, exclude=exclude)
        example = self.compile_example(schema=schema, tag=TAG, mode=mode, exclude=exclude)
        # nodes = ", ".join(self.to_dict(mode=mode).keys())
        constraints = [LANGUAGE_CONSTRAINT, FORMAT_CONSTRAINT]
        constraint = "\n".join(constraints)

        prompt = template.format(
            context=context,
            example=example,
            instruction=instruction,
            constraint=constraint,
        )
        return prompt
    
    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand the xml format.
        """
        field_names = self.get_field_names()
        # Construct the example using the field names
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")

        # Join all examples into a single string
        example_str = "\n".join(examples)
        # Add the example to the context
        context += f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""
        return context  
    
    async def code_fill(
        self, context: str, system_msg: List[str], function_name: Optional[str] = None, timeout: int = USE_CONFIG_TIMEOUT
    ) -> Dict[str, str]:
        """
        Fill CodeBlock Using ``` ```
        """
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, system_msgs=system_msg, timeout=timeout)
        extracted_code = sanitize(code=content, entrypoint=function_name)
        result = {field_name: extracted_code}
        return result
    
    async def single_fill(self, context: str, system_msg: List[str], images: Optional[Union[str, list[str]]] = None) -> Dict[str, str]:
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, system_msgs=system_msg, images=images)
        result = {field_name: content}
        return result

    async def xml_fill(self, context: str, system_msg: List[str], images: Optional[Union[str, list[str]]] = None) -> Dict[str, Any]:
        """
        Fill context with XML tags and convert according to field types, including string, integer, boolean, list and dict types.
        Support nested XML structure.
        """
        def extract_nested_xml(content, node):
            """Recursively extract data from nested XML structure"""
            extracted = {}
            
            if node.children:
                # If there are children, parse nested structure
                for child_key, child_node in node.children.items():
                    pattern = rf"<{child_key}>(.*?)</{child_key}>"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        raw_value = match.group(1).strip()
                        
                        # If child has children, recursive parse
                        if child_node.children:
                            extracted[child_key] = extract_nested_xml(raw_value, child_node)
                        else:
                            # Parse simple field
                            field_type = child_node.expected_type
                            extracted[child_key] = self._convert_field_value(raw_value, field_type)
            else:
                # If there are no children, parse single field
                field_type = node.expected_type
                extracted = self._convert_field_value(content, field_type)
                
            return extracted
        
        content = await self.llm.aask(context, system_msgs=system_msg, images=images)
        
        # Extract from root node
        if self.children:
            return extract_nested_xml(content, self)
        else:
            # Single field case
            field_name = self.get_field_name()
            pattern = rf"<{field_name}>(.*?)</{field_name}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                return {field_name: self._convert_field_value(raw_value, self.expected_type)}
            return {}
    
    def _convert_field_value(self, raw_value: str, field_type: Type) -> Any:
        """Convert raw string value to appropriate type"""
        import json
        import typing
        
        if field_type == str:
            return raw_value
        elif field_type == int:
            try:
                return int(raw_value)
            except ValueError:
                return 0
        elif field_type == bool:
            return raw_value.lower() in ("true", "yes", "1", "on", "True")
        elif field_type == list or (hasattr(field_type, '__origin__') and field_type.__origin__ is list):
            try:
                # Try to parse as JSON first (safer than eval)
                result = json.loads(raw_value)
                if not isinstance(result, list):
                    raise ValueError
                return result
            except (json.JSONDecodeError, ValueError):
                try:
                    # Fallback to eval for backward compatibility
                    result = eval(raw_value)
                    if not isinstance(result, list):
                        raise ValueError
                    return result
                except:
                    return []
        elif field_type == dict or (hasattr(field_type, '__origin__') and field_type.__origin__ is dict):
            try:
                # Try to parse as JSON first (safer than eval)
                result = json.loads(raw_value)
                if not isinstance(result, dict):
                    raise ValueError
                return result
            except (json.JSONDecodeError, ValueError):
                try:
                    # Fallback to eval for backward compatibility
                    result = eval(raw_value)
                    if not isinstance(result, dict):
                        raise ValueError
                    return result
                except:
                    return {}
        else:
            return raw_value
    
    async def simple_fill(
        self, schema, mode, system_msg: List[str], images: Optional[Union[str, list[str]]] = None, timeout=USE_CONFIG_TIMEOUT, exclude=None
    ):
        prompt = self.compile(context=self.context, schema=schema, mode=mode, exclude=exclude)
        if schema != "raw":
            mapping = self.get_mapping(mode, exclude=exclude)
            class_name = f"{self.key}_AN"
            content, scontent = await self._aask_v1(
                prompt, class_name, mapping, images=images, schema=schema, timeout=timeout, system_msgs=system_msg
            )
            self.content = content
            self.instruct_content = scontent
        else:
            self.content = await self.llm.aask(prompt, system_msgs=system_msg)
            self.instruct_content = None

        return self
    
    def get_field_name(self):
        """
        Get the field name from the Pydantic model associated with this ActionNode.
        """
        model_class = self.create_class()
        fields = model_class.model_fields

        # Assuming there's only one field in the model
        if len(fields) == 1:
            return next(iter(fields))

        # If there are multiple fields, we might want to use self.key to find the right one
        return self.key
    
    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _aask_v1(
        self,
        prompt: str,
        output_class_name: str,
        output_data_mapping: dict,
        images: Optional[Union[str, list[str]]] = None,
        system_msgs: Optional[list[str]] = None,
        schema="markdown",  # compatible to original format
        timeout=USE_CONFIG_TIMEOUT,
    ) -> (str, BaseModel):
        """Use ActionOutput to wrap the output of aask"""
        content = await self.llm.aask(prompt, system_msgs, images=images, timeout=timeout)
        logger.debug(f"llm raw output:\n{content}")
        output_class = self.create_model_class(output_class_name, output_data_mapping)

        if schema == "json":
            raise NotImplementedError("json schema is not supported")
            # parsed_data = llm_output_postprocess(
            #     output=content, schema=output_class.model_json_schema(), req_key=f"[/{TAG}]"
            # )
        else:  # using markdown parser
            parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)

        logger.debug(f"parsed_data:\n{parsed_data}")
        instruct_content = output_class(**parsed_data)
        return content, instruct_content
    
    async def fill(
        self,
        *,
        context,
        llm,
        schema="json",
        mode="auto",
        strgy="simple",
        images: Optional[Union[str, list[str]]] = None,
        timeout=USE_CONFIG_TIMEOUT,
        exclude=[],
        function_name: Optional[str] = None,
        system_msg: List[str] = [],
    ):
        """Fill the node(s) with mode.

        :param context: Everything we should know when filling node.
        :param llm: Large Language Model with pre-defined system message.
        :param schema: json/markdown, determine example and output format.
         - raw: free form text
         - json: it's easy to open source LLM with json format
         - markdown: when generating code, markdown is always better
        :param mode: auto/children/root
         - auto: automated fill children's nodes and gather outputs, if no children, fill itself
         - children: fill children's nodes and gather outputs
         - root: fill root's node and gather output
        :param strgy: simple/complex
         - simple: run only once
         - complex: run each node
        :param images: the list of image url or base64 for gpt4-v
        :param timeout: Timeout for llm invocation.
        :param exclude: The keys of ActionNode to exclude.
        :return: self
        """
        self.set_llm(llm)
        self.set_context(context)
        if self.schema:
            schema = self.schema

        if mode == FillMode.CODE_FILL.value:
            result = await self.code_fill(context, system_msg=system_msg, function_name=function_name, timeout=timeout)
            self.instruct_content = self.create_class()(**result)
            return self

        elif mode == FillMode.XML_FILL.value:
            context = self.xml_compile(context=self.context)
            result = await self.xml_fill(context, system_msg=system_msg, images=images)
            self.instruct_content = self.create_class()(**result)
            return self

        elif mode == FillMode.SINGLE_FILL.value:
            result = await self.single_fill(context, system_msg=system_msg, images=images)
            self.instruct_content = self.create_class()(**result)
            return self

        if strgy == "simple":
            return await self.simple_fill(schema=schema, mode=mode, system_msg=system_msg, images=images, timeout=timeout, exclude=exclude)
        elif strgy == "complex":
            # 这里隐式假设了拥有children
            tmp = {}
            for _, i in self.children.items():
                if exclude and i.key in exclude:
                    continue
                child = await i.simple_fill(schema=schema, mode=mode, system_msg=system_msg, images=images, timeout=timeout, exclude=exclude)
                tmp.update(child.instruct_content.model_dump())
            cls = self._create_children_class()
            self.instruct_content = cls(**tmp)
            return self
        
    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], key: str = None):
        """
        Creates an ActionNode tree from a Pydantic model.

        Args:
            model (Type[BaseModel]): The Pydantic model to convert.

        Returns:
            ActionNode: The root node of the created ActionNode tree.
        """
        key = key or model.__name__
        root_node = cls(key=key, expected_type=Type[model], instruction="", example="")

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description
            default = field_info.default

            # Recursively handle nested models if needed
            if not isinstance(field_type, typing._GenericAlias) and issubclass(field_type, BaseModel):
                child_node = cls.from_pydantic(field_type, key=field_name)
            else:
                child_node = cls(key=field_name, expected_type=field_type, instruction=description, example=default)

            root_node.add_child(child_node)

        return root_node

    
    @staticmethod
    def is_optional_type(tp) -> bool:
        """Return True if `tp` is `typing.Optional[...]`"""
        if typing.get_origin(tp) is Union:
            args = typing.get_args(tp)
            non_none_types = [arg for arg in args if arg is not type(None)]
            return len(non_none_types) == 1 and len(args) == 2
        return False

from pydantic import BaseModel, Field
import asyncio
from provider.llm_provider_registry import create_llm_instance
from utils.cost_manager import CostManager
from configs.models_config import ModelsConfig

class AnswerOp(BaseModel):
    code: str = Field(default="", description="Your solution for this problem")
    explanation: str = Field(default="", description="Your explanation for this problem")

class GenerateOp(BaseModel):
    response: AnswerOp = Field(default_factory=lambda: AnswerOp(), description="Your solution for this problem")
    # response: str = Field(default="", description="Your solution for this problem")
    
async def process():
    llm = create_llm_instance(ModelsConfig.default().get("gpt-4o-mini"))
    llm.cost_manager = CostManager()
    tmp = await ActionNode.from_pydantic(GenerateOp).fill(
        context="Write a simple python code to display using matplot", mode="xml_fill", llm=llm,
        system_msg=["Provide both code and explain about it"]
    )
    print(tmp.instruct_content.model_dump())

if __name__ == '__main__':
    asyncio.run(process())