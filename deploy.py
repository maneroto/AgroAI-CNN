from keras.models import load_model
import tensorflow as tf

from config.settings import MODEL_NAME

model = load_model(f'{MODEL_NAME}.h5')

frozen_model = tf.function(lambda x: model(x))
frozen_model = frozen_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)

frozen_graph = tf.graph_util.convert_variables_to_constants_v2(frozen_model.graph)
tf.io.write_graph(
  graph_or_graph_def=frozen_graph,
  logdir='.',
  name=f'{MODEL_NAME}.pb',
  as_text=False
)