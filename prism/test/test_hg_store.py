import os

from utils.constants import ROOT_PATH

from prism.settings import prism_embedding_model
from prism.hg_store.store import EmbeddingStore

if __name__ == "__main__":
    store = EmbeddingStore(
        prism_embedding_model,
        os.path.join(ROOT_PATH, "cache", "hg_store"),
        32,
        "hg_store"
    )
    print(store.get_row("dima806/67_cat_breeds_image_detection"))