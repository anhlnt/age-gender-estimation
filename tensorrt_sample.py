import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np


def main():
    input_saved_model_dir = "pretrained_models/EfficientNetB3_224_weights.26-3.15"
    output_saved_model_dir = "pretrained_models/TensorRT/EfficientNetB3_224_weights.26-3.15"


    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(
        maximum_cached_engines=100)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params)
    converter.convert()
    print("[LOG] Successfully converted")
    def my_input_fn():
        # Input for a single inference call, for a network that has two input tensors:
        inp1 = np.random.normal(size=(8, 224, 224, 3)).astype(np.float32)
        # inp2 = np.random.normal(size=(8, 16, 16, 3)).astype(np.float32)
        yield inp1
    # converter.build(input_fn=my_input_fn)
    # print("[LOG] Successfully build")
    converter.save(output_saved_model_dir)
    print("[LOG] Successfully saved TensorRT Model")


    saved_model_loaded = tf.saved_model.load(
        output_saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        graph_func)
    output = frozen_func(input_data)[0].numpy()

if __name__ == "__main__":
    main()