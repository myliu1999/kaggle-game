#!/usr/bin/env python3

import codecs
import csv

import pandas as pd

OUTPUT_FILENAME = 'out.ans'

train_data = 'data/train.csv'
test_data = 'data/test.csv'
toxic_data = 'toxic.txt'

csv_header = [
    'id',
    'comment_text',
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

toxic_words = []


def get_score(data):
    score = 0
    for i in range(2, 2 + 6):
        score += int(data[csv_header[i]])
    return score


def read_csv(filename):
    return csv.DictReader(codecs.open(filename, 'r', encoding='utf-8'))


def predict_toxic(data):
    if any(word in data.lower() for word in toxic_words):
        return 1.0
    return 0.0


def write_out_answer(output_filename):
    df = pd.read_csv(test_data, na_filter=False)

    f = open(output_filename, 'w')
    f.write('%s,' % csv_header[0])
    for item in csv_header[2:-1]:
        f.write('%s,' % item)
    f.write('%s\n' % csv_header[-1])

    for index, row in df.iterrows():
        f.write('%s,' % row[csv_header[0]])
        for i in range(5):
            f.write('%d,' % predict_toxic(row[csv_header[1]]))
        f.write('%d\n' % predict_toxic(row[csv_header[1]]))

    f.close()


def prepare_toxic():
    t = open(toxic_data, 'r')

    for i in t:
        toxic_words.append(i.strip('\n'))

    t.close()


def main():
    prepare_toxic()

    train_list = list(read_csv(train_data))
    # test_list = list(read_csv(test_data))
    train_list_size = len(train_list)
    # test_list_size = len(test_list)
    count = 0
    miss_count = 0
    for i in range(train_list_size):
        if any(word in train_list[i]['comment_text'].lower() for word in toxic_words):
            count += 1

            score = get_score(train_list[i])
            if score == 0:
                miss_count += 1

    print('count = %d' % (count))
    print('miss_count = %d' % (miss_count))
    print('train_list_size = %d' % (train_list_size))

    write_out_answer(OUTPUT_FILENAME)


if __name__ == '__main__':
    main()
