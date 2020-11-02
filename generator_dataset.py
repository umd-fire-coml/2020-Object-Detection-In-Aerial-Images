# %%
import tensorflow as tf
from data_generator import DOTASequence
from augmentation import augment
import time
# %%
class DOTASequenceDataset(tf.data.Dataset):
    def __new__(cls, data_generator):
        return tf.data.Dataset.from_generator(
            data_generator,
            output_types=tf.dtypes.float64)
# %%
def benchmark(dataset, epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(epochs):
        for sample in dataset:
            dataset.__getitem__(sample)
    print(time.perf_counter() - start_time)

# %%
generator = iter(DOTASequence('data/train/images', 'data/train/annotations_hbb', augment))
benchmark(tf.data.Dataset.range(2).interleave(DOTASequenceDataset(generator), nunum_parallel_calls=tf.data.experimental.AUTOTUNE))

# To do: dataset.prefetch(tf.data.experimental.AUTOTUNE), 
# tf.data.range(2).interleave(dataset, nunum_parallel_calls=tf.data.experimental.AUTOTUNE) 
# then the augment stuff and .map(augment(args), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
# Do cache if have time.

# %%
