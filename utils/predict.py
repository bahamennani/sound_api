import numpy as np
import tensorflow as tf

interpreter_essence = tf.lite.Interpreter(model_path="models/cnn_model_essence.tflite")
interpreter_diesel = tf.lite.Interpreter(model_path="models/cnn_model_diesel.tflite")
interpreter_essence.allocate_tensors()
interpreter_diesel.allocate_tensors()

labels_essence = ['Healthy engine', "Consommation d'huile", 'Ralenti instable', 'Raté d’allumage']
labels_diesel = ['Healthy engine', 'Damaged engine']

def predict_fault(engine_type, spectrogram_input):
    if engine_type == 'essence':
        interpreter = interpreter_essence
        labels = labels_essence
    elif engine_type == 'diesel':
        interpreter = interpreter_diesel
        labels = labels_diesel
    else:
        raise ValueError("Invalid engine type")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], spectrogram_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = output_data[0]
    label_index = np.argmax(prediction)
    confidence = prediction[label_index]

    return labels[label_index], confidence