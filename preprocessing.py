import tensorflow as tf

def decode_and_resize(image_bytes, size=224):
    """Shared by training and serving: decode to RGB, resize, keep 0-255 pixels."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)