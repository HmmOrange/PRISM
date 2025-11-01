from utils.cost_manager import CostManager, TokenCostManager, FireworksCostManager
from utils.constants import CONFIG_PATH
from provider.embedding_provider_registry import create_embedding_instance
from configs.models_config import ModelsConfig

import yaml

cost_manager = CostManager()
token_cost_manager = TokenCostManager()
fireworks_cost_manager = FireworksCostManager()

config = yaml.load(open(f"{CONFIG_PATH}/config.yaml", "r"), Loader=yaml.FullLoader)

embedding_model_config = ModelsConfig.default().get_embedding_config(config["prism"]["embedding_model"])
prism_embedding_model = create_embedding_instance(embedding_model_config)
prism_embedding_model.cost_manager = cost_manager

# retriever_type = config["prism"]["type"]
glue_coder_max_react_loop = config["prism"]["glue_coder_max_react_loop"]
ml_coder_max_react_loop = config["prism"]["ml_coder_max_react_loop"]
retriever_top_k = config["prism"]["top_k"]
embedding_model = config["prism"]["embedding_model"]
model_pool_size = config["prism"]["model_pool_size"]
hg_store_name = f"hg_store_{embedding_model.replace('/', '_').replace('-', '_')}_{model_pool_size}"