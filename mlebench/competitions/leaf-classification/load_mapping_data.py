import os
from typing import Optional
import re
import pandas as pd

def load_train_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   train_label_path = os.path.join(public_path, "train.csv")
   train_label_df = pd.read_csv(train_label_path)
   label_lookup = train_label_df.set_index("id").to_dict(orient="index")
   train_path = os.path.join(public_path, "train")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(train_path):
      file_name, _ = os.path.splitext(file)
      input_data = {
         "image_paths": [os.path.join(train_path, file)],
         "input_id": file_name,
         "label": {
             "species": label_lookup[int(file_name)]["species"]
         }
      }
      data.append(input_data)
   return data, train_label_df


def load_test_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   private_path = os.path.join(path, "prepared", "private")
   test_label_path = os.path.join(private_path, "answers.csv")
   test_label_df = pd.read_csv(test_label_path)
   test_path = os.path.join(public_path, "test")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(test_path):
        file_name, _ = os.path.splitext(file)
        input_data = {
            "image_paths": [os.path.join(test_path, file)],
            "input_id": file_name,
            "fields": {
                "id": file_name
            },
            "predict_keys": ["prediction"]
        }
        data.append(input_data)
   return data, test_label_df

def load_lite_test_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   private_lite_path = os.path.join(path, "prepared_lite", "private")
   test_label_path = os.path.join(private_lite_path, "answers.csv")
   test_label_df = pd.read_csv(test_label_path)
   test_path = os.path.join(public_path, "test")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(test_path):
      if file.endswith(tuple(image_extensions)):
         file_name, _ = os.path.splitext(file)
         # Only include files that are in the lite test set
         if int(file_name) in test_label_df["id"].values:
            input_data = {
               "image_paths": [os.path.join(test_path, file)],
               "input_id": file_name,
               "fields": {
                  "id": file_name
               },
               "predict_keys": ["prediction"]
            }
            data.append(input_data)
   return data, test_label_df

SPECIE_LIST = [
    "Acer_Capillipes", "Acer_Circinatum", "Acer_Mono", "Acer_Opalus", "Acer_Palmatum",
    "Acer_Pictum", "Acer_Platanoids", "Acer_Rubrum", "Acer_Rufinerve", "Acer_Saccharinum",
    "Alnus_Cordata", "Alnus_Maximowiczii", "Alnus_Rubra", "Alnus_Sieboldiana", "Alnus_Viridis",
    "Arundinaria_Simonii", "Betula_Austrosinensis", "Betula_Pendula", "Callicarpa_Bodinieri",
    "Castanea_Sativa", "Celtis_Koraiensis", "Cercis_Siliquastrum", "Cornus_Chinensis",
    "Cornus_Controversa", "Cornus_Macrophylla", "Cotinus_Coggygria", "Crataegus_Monogyna",
    "Cytisus_Battandieri", "Eucalyptus_Glaucescens", "Eucalyptus_Neglecta", "Eucalyptus_Urnigera",
    "Fagus_Sylvatica", "Ginkgo_Biloba", "Ilex_Aquifolium", "Ilex_Cornuta",
    "Liquidambar_Styraciflua", "Liriodendron_Tulipifera", "Lithocarpus_Cleistocarpus",
    "Lithocarpus_Edulis", "Magnolia_Heptapeta", "Magnolia_Salicifolia", "Morus_Nigra",
    "Olea_Europaea", "Phildelphus", "Populus_Adenopoda", "Populus_Grandidentata",
    "Populus_Nigra", "Prunus_Avium", "Prunus_X_Shmittii", "Pterocarya_Stenoptera",
    "Quercus_Afares", "Quercus_Agrifolia", "Quercus_Alnifolia", "Quercus_Brantii",
    "Quercus_Canariensis", "Quercus_Castaneifolia", "Quercus_Cerris", "Quercus_Chrysolepis",
    "Quercus_Coccifera", "Quercus_Coccinea", "Quercus_Crassifolia", "Quercus_Crassipes",
    "Quercus_Dolicholepis", "Quercus_Ellipsoidalis", "Quercus_Greggii", "Quercus_Hartwissiana",
    "Quercus_Ilex", "Quercus_Imbricaria", "Quercus_Infectoria_sub", "Quercus_Kewensis",
    "Quercus_Nigra", "Quercus_Palustris", "Quercus_Phellos", "Quercus_Phillyraeoides",
    "Quercus_Pontica", "Quercus_Pubescens", "Quercus_Pyrenaica", "Quercus_Rhysophylla",
    "Quercus_Rubra", "Quercus_Semecarpifolia", "Quercus_Shumardii", "Quercus_Suber",
    "Quercus_Texana", "Quercus_Trojana", "Quercus_Variabilis", "Quercus_Vulcanica",
    "Quercus_x_Hispanica", "Quercus_x_Turneri", "Rhododendron_x_Russellianum",
    "Salix_Fragilis", "Salix_Intergra", "Sorbus_Aria", "Tilia_Oliveri", "Tilia_Platyphyllos",
    "Tilia_Tomentosa", "Ulmus_Bergmanniana", "Viburnum_Tinus", "Viburnum_x_Rhytidophylloides",
    "Zelkova_Serrata"
]

def normalize(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for comparison."""
    return re.sub(r'[^a-z0-9]', '', text.lower())
 
def resolve_specie(specie: str) -> Optional[str]:
   try:
      normalized_specie = normalize(specie)
      
      for s in SPECIE_LIST:
         if normalize(s) == normalized_specie:
               return s
      
      for s in SPECIE_LIST:
         if normalized_specie in normalize(s) or normalize(s) in normalized_specie:
               return s
   except:
      return None
    
   return None

def mapping(predictions: pd.DataFrame, path: str) -> pd.DataFrame:
   submissions = pd.DataFrame(0, index=predictions.index, columns=SPECIE_LIST)
   for i, specie in enumerate(predictions["prediction"]):
      chosen_specie = resolve_specie(specie)
      if chosen_specie is not None:
         submissions.at[i, chosen_specie] = 1
   
   submissions.insert(0, "id", predictions["id"])
   
   return submissions