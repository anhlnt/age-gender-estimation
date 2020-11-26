import tensorflow as tf
from src.factory import get_model
from omegaconf import OmegaConf
from pathlib import Path



def getModel():
    weight_file = "pretrained_models/EfficientNetB3_224_weights.26-3.15.hdf5"
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    print('model_name: ', model_name, 'img_size: ', img_size)
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)
    return model

def saveModel(model, path):
    tf.saved_model.save(model, path)

def main():
    model = getModel()
    savePath = 'pretrained_models/EfficientNetB3_224_weights.26-3.15'
    saveModel(model, savePath)

if __name__ == "__main__":
    main()