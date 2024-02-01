import tensorflow as tf
from transformer import *
from pickle import load


tags = [
    "informal",
    "android",
    "azae",
    "chat",
    "choding",
    "emoticon",
    "enfp",
    "gentle",
    "halbae",
    "halmae",
    "joongding",
    "king",
    "naruto",
    "seonbi",
    "sosim",
    "translator"
]

tokenizer = load(open(f".\\model\\tokenizer.pkl", "rb"))

START_TOKEN, END_TOKEN, VOCAB_SIZE = SEV(tokenizer)

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(f".\\model\\model.ckpt")


while True:
    user = input(" >>> ")
    out = predict(user, tokenizer, model, START_TOKEN, END_TOKEN, MAX_LENGTH)
    print("token: "+ str(tokenizer.encode(user)))
    print("token: "+ str([tokenizer.decode([c]) for c in tokenizer.encode(user)]))

    print("translation: " + out)