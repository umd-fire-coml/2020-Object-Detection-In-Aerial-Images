# %%
import tensorflow as tf
from data_generator import DOTASequence
from augmentation import augment
import time
# %%
class DOTASequenceDataset(tf.data.Dataset):
    def _generator(data_generator):
        dataset = tf.data.Dataset.from_generator(data_generator)
    def __new__(cls, data_generator):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float64,
            args=(data_generator,)
        )
# %%
def benchmark(dataset, epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(epochs):
        for sample in dataset:
            time.sleep(.01)
    print(time.perf_counter() - start_time)

# %%
generator = DOTASequence('data/train/images', 'data/train/annotations_hbb')
benchmark(DOTASequenceDataset(generator))

# To do: dataset.prefetch(tf.data.experimental.AUTOTUNE), 
# tf.data.range(2).interleave(dataset, nunum_parallel_calls=tf.data.experimental.AUTOTUNE) 
# then the augment stuff and .map(augment(args), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
# Do cache if have time.

# To ask: Current issue: data_generator is Keras Sequence but we want a python generator. Is there way to fix?
# Quick google search had nothing.
# 2. Using getitem for a series of numpy arrays, but tf.data.Dataset.from_tensor_slices doesn't accept that. Why? Feasible?

# %%
