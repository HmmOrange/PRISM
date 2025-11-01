import re
import os
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
      if file.endswith(tuple(image_extensions)):
         file_name = file.split(".")[0]
         input_data = {
            "image_paths": [os.path.join(train_path, file)],
            "label": {
               "breed": label_lookup[file_name]["breed"]
            },
            "input_id": file_name
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
      file_name = file.split(".")[0]
      if file.endswith(tuple(image_extensions)):
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
         file_name = file.split(".")[0]
         # Only include files that are in the lite test set
         if file_name in test_label_df["id"].values:
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

BREED_LIST = ["affenpinscher","afghan_hound","african_hunting_dog","airedale","american_staffordshire_terrier","appenzeller","australian_terrier","basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog","black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick","border_collie","border_terrier","borzoi","boston_bull","bouvier_des_flandres","boxer","brabancon_griffon","briard","brittany_spaniel","bull_mastiff","cairn","cardigan","chesapeake_bay_retriever","chihuahua","chow","clumber","cocker_spaniel","collie","curly-coated_retriever","dandie_dinmont","dhole","dingo","doberman","english_foxhound","english_setter","english_springer","entlebucher","eskimo_dog","flat-coated_retriever","french_bulldog","german_shepherd","german_short-haired_pointer","giant_schnauzer","golden_retriever","gordon_setter","great_dane","great_pyrenees","greater_swiss_mountain_dog","groenendael","ibizan_hound","irish_setter","irish_terrier","irish_water_spaniel","irish_wolfhound","italian_greyhound","japanese_spaniel","keeshond","kelpie","kerry_blue_terrier","komondor","kuvasz","labrador_retriever","lakeland_terrier","leonberg","lhasa","malamute","malinois","maltese_dog","mexican_hairless","miniature_pinscher","miniature_poodle","miniature_schnauzer","newfoundland","norfolk_terrier","norwegian_elkhound","norwich_terrier","old_english_sheepdog","otterhound","papillon","pekinese","pembroke","pomeranian","pug","redbone","rhodesian_ridgeback","rottweiler","saint_bernard","saluki","samoyed","schipperke","scotch_terrier","scottish_deerhound","sealyham_terrier","shetland_sheepdog","shih-tzu","siberian_husky","silky_terrier","soft-coated_wheaten_terrier","staffordshire_bullterrier","standard_poodle","standard_schnauzer","sussex_spaniel","tibetan_mastiff","tibetan_terrier","toy_poodle","toy_terrier","vizsla","walker_hound","weimaraner","welsh_springer_spaniel","west_highland_white_terrier","whippet","wire-haired_fox_terrier","yorkshire_terrier"]

def normalize(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for comparison."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def resolve_breed(breed: str) -> str:
   normalized_breed = normalize(breed)
   
   for b in BREED_LIST:
      if normalize(b) == normalized_breed:
         return b

   for b in BREED_LIST:
      if normalized_breed in normalize(b) or normalize(b) in normalized_breed:
         return b
   
   return None

def mapping(predictions: pd.DataFrame, path: str):
   submissions = pd.DataFrame(0, index=predictions.index, columns=BREED_LIST)
   
   for i, breed in enumerate(predictions["prediction"]):
      chosen_breed = resolve_breed(breed)
      if chosen_breed is not None:
         submissions.at[i, chosen_breed] = 1
   
   submissions.insert(0, "id", predictions["id"])
   
   return submissions
