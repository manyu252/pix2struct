import onnxruntime
import requests
from PIL import Image

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Load the encoder ONNX model
encoder_model_path = 'google/pix2struct-docvqa-base/encoder_model.onnx'
encoder_session = onnxruntime.InferenceSession(encoder_model_path)

# Load the decoder ONNX model
decoder_model_path = 'google/pix2struct-docvqa-base/decoder_model.onnx'
decoder_session = onnxruntime.InferenceSession(decoder_model_path)

# Replace this with your actual input data
input_data = image

# Run inference on the encoder model
encoded_representation = encoder_session.run(None, {'input_name': input_data})[0]

# Run inference on the decoder model
output_data = decoder_session.run(None, {'input_name': encoded_representation})[0]

# Replace this with your post-processing logic
# decoded_text = post_process_output(output_data)
