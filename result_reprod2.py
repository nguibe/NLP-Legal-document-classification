# Assuming the dataset is already loaded as a `Dataset` object
# The label_index is a dictionary mapping label names to integer indices

from datasets import load_dataset

# Example loading Eurlex dataset
dataset = load_dataset('eurlex', split='train')  # Use your dataset here
label_index = {label: idx for idx, label in enumerate(sorted(set(dataset['labels'])))}

train_dataset = load_dataset('multi_eurlex', language='all_languages',
                                 languages=['en'],
                                 label_level='level_3', split='train')
eval_dataset = load_dataset('multi_eurlex', language='all_languages',
                                languages=['fr','de','pl','fi'], label_level='level_3')

# Initialize SampleGenerator
batch_size = 8
sample_generator = SampleGenerator(
    dataset=dataset,
    label_index=label_index,
    bert_model_path='xlm-roberta-base',
    lang='en',  # Set the language of choice
    multilingual_train=False,
    batch_size=batch_size,
    shuffle=True,
    max_document_length=512
)

# Define model and training setup (example with TensorFlow/Keras)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(batch_size, 512)),  # Adjust input shape
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(label_index), activation='sigmoid')  # Sigmoid for multi-label
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model using SampleGenerator as the data source
model.fit(sample_generator, epochs=3)
