from scripts.model_inference import model_inference

def main():
    res = model_inference(model_id="microsoft/git-base-msrvtt-qa", input_data={"image": "tasks/graph-level/audio-image-consistency/validation/inputs/1/image.jpg"}, hosted_on="local", task="image-to-text")
    print(res)
if __name__ == '__main__':
    main()