import tensorflow as tf
import tensorflow_datasets as tfds
import re

def CleanEnding(sentence:str):
    # 구두점 띄어쓰기
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()

    return sentence

def tokenize_and_filter(tokenizer, inputs:list, outputs:list, START_TOKEN:list, END_TOKEN:list, MAX_LENGTH:int):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
    
    return tokenized_inputs, tokenized_outputs

def SEV(tokenizer) -> tuple[list, list, int]:

    '''
    return START_TOKEN, END_TOKEN, VOCAB_SIZE
    '''

    START_TOKEN = [tokenizer.vocab_size]
    END_TOKEN = [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2

    return START_TOKEN, END_TOKEN, VOCAB_SIZE

def create_tokenizer(corpus:list):
    return tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=2**13)

def create_dataset(questions, answers, BUFFER_SIZE, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
        },
        {
            'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
        },
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def evaluate(sentence, model, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence, tokenizer, model, START_TOKEN, END_TOKEN, MAX_LENGTH):
    prediction = evaluate(sentence, model, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence
  
  
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence