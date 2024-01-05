from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
''' 
In this tutorial we create a simple ANN using the functional Keras API. As 
models grow more complex, it is useful to see a representation of the "mental
model" we've created. A functional model is an explicit data structure, which 
makes it possible to explore the layers and reuse previous graph nodes. To 
visualize this graph (or better said the topology) of the model, we will use 
the plot_model() utility.
'''

inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
#model.summary()

## Exampple
vocabulary_size = 10000 
num_tags = 100 
num_departments = 4 
  
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")
 
features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)
 
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department")(features)
 
model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])
  
num_samples = 1280 
  
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
 
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))
 
model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data])

keras.utils.plot_model(model, "ticket_classifier.png")

# Retrieving inputs or outputs of a layer
model.layers
model.layers[3].input

# Creating a new model by reusing intermediate layer outputs
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
  
new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty])
keras.utils.plot_model(
    new_model, "updated_ticket_classifier.png", show_shapes=True)