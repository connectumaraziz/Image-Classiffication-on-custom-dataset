import tensorflow as tf
import gradio as gr
model = tf.keras.models.load_model('../Path_to_Output_model_weight/..')

classes = []
with open("classes.txt", 'r') as f:
  classes = list(map(lambda x: x.strip(), f.readlines()))

def classify_image(inp):
  inp = inp.reshape((-1, 299, 299, 3))
  inp = tf.keras.applications.xception.preprocess_input(inp)
  prediction = model.predict(inp).flatten()
  confidences = {classes[i]: float(prediction[i]) for i in range(len(prediction))}
  return confidences


gr.Interface(fn=classify_image,
             inputs=gr.inputs.Image(shape=(299, 299)),
             outputs=gr.outputs.Label(num_top_classes=3)).launch()
