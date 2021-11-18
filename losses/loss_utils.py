import tensorflow as tf

# https://stackoverflow.com/questions/37086098/does-tensorflow-map-fn-support-taking-more-than-one-tensor
def my_map(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out