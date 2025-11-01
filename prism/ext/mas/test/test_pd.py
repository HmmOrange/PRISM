import pandas as pd

if __name__ == "__main__":
    res = [{
        "image_paths": ["321", "123"]
    }]
    res_pd = pd.DataFrame(res)
    print(res_pd.iloc[0].to_dict())
    for image_path in res_pd.iloc[0]["image_paths"]:
        print(image_path)