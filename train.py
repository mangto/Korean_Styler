import tensorflow as tf
from transformer import *
import pandas as pd
from pickle import dump, load, HIGHEST_PROTOCOL

print("[!] End Loading")


train_data = pd.read_csv(".\\dataset.tsv", sep="	")
questions = [CleanEnding(c) for c in train_data['original']]
answers = [CleanEnding(c) for c in train_data['styled']]

corpus = questions+answers
tokenizer = create_tokenizer(corpus)
with open(".\\model\\tokenizer.pkl", "wb") as file:
    dump(tokenizer, file, protocol=HIGHEST_PROTOCOL)


START_TOKEN, END_TOKEN, VOCAB_SIZE = SEV(tokenizer)

questions, answers = tokenize_and_filter(tokenizer, 
                                         questions, answers,
                                         START_TOKEN, END_TOKEN, MAX_LENGTH)

dataset = create_dataset(questions, answers, BUFFER_SIZE, BATCH_SIZE)

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f".\\model\\model.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model.fit(dataset,epochs=EPOCHS, callbacks=[cp_callback])