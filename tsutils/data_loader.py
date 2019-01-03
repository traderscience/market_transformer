# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
import pandas as pd
from ts_FeatureCoding import Feature_Coding

DATA_DIR = "data/events"

class DataLoader():
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.data_file = args.data_file
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.max_records = args.max_records
        self.encoding=args.input_encoding

        self.featureCodes = Feature_Coding()
        self.nfeatures = self.featureCodes.nfeatures


        input_file = os.path.join(self.data_dir, self.data_file)
        print("reading text file")
        self.loadcsv(input_file)


    def preparedata(self):
        vocab_file = os.path.join(self.data_dir, "vocab.pkl")
        tensor_file = os.path.join(self.data_dir, "data.npy")


        # Let's not read vocab and data from file. We may change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("building vocabulary files...")
            self.preprocess(vocab_file, tensor_file, self.encoding)
        else:
            print("loading preprocessed files...")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        #string = re.sub(r"_", " Period_", string)
        string = re.sub(r",", "_", string)
        string = re.sub(r"VS_15,Neutral", "\.", string)
        return string
        #string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        #string = re.sub(r"\'s", " \'s", string)
        #string = re.sub(r"\'ve", " \'ve", string)
        #string = re.sub(r"n\'t", " n\'t", string)
        #string = re.sub(r"\'re", " \'re", string)
        #string = re.sub(r"\'d", " \'d", string)
        #string = re.sub(r"\'ll", " \'ll", string)
        #string = re.sub(r"!", " ! ", string)
        #string = re.sub(r"\(", " \( ", string)
        #string = re.sub(r"\)", " \) ", string)
        #string = re.sub(r"\?", " \? ", string)
        #string = re.sub(r"\s{2,}", " ", string)
        #return string.strip().lower()

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def loadcsv(self, input_file):
        columns= self.featureCodes.featuresAll
        nread = 100000
        skip_rows = 0
        max_records = self.max_records
        self.raw_df = pd.DataFrame(columns=columns)

        reader = pd.read_csv(input_file, iterator=True, chunksize=nread,
                             header=0, names=columns, index_col=False,
                             na_values='NA', skip_blank_lines=True,
                             skipinitialspace=True, infer_datetime_format=False,
                             parse_dates=False, skiprows=skip_rows)
        do_more = True
        total_read = 0
        dailyRowSeen = False

        for csvrows in reader:
            if csvrows.shape[0] == 0:
                doMore = False
                break
            # convert TimeStamp column to a datatime
            csvrows['TimeStamp'] = pd.to_datetime(csvrows['TimeStamp'], format='%Y/%m/%dT%H:%M:%S')

            # raw_df = raw_df.append(csvrows, ignore_index=True)
            self.raw_df = pd.concat([self.raw_df, csvrows], axis=0, copy=False, ignore_index=True)
            skip_rows += nread
            total_read += nread
            print('Records read:', total_read, self.raw_df.shape)
            if max_records > 0 and total_read >= max_records:
                doMore = False
                break


        print('Total Records read:', total_read, ' Saved:', self.raw_df.shape)

        self.raw_df.columns = columns
        self.raw_df.set_index('TimeStamp')

        """
        # extract the event TypeCode
        self.raw_df['TypeCode'] = self.raw_df['Type'].str.split('_').str[0]

        # extract the Direction code
        self.raw_df['Dir'] = self.raw_df['TypeCode'].str[-1:]
        self.raw_df['Period'] = self.raw_df['Type'].str.split('_').str[1]

        # map the Period (D,60,15,5,1) to int PeriodCode (1440,60,15,5,1)
        try:
            self.raw_df['TypeCodeNum'] = self.raw_df['TypeCode'].map(self.featureCodes.eventCodeDict).astype('int32')
            self.raw_df['PeriodCode'] = self.raw_df['Period'].map(self.featureCodes.periodCodeDict).astype('int32')
        except RuntimeError as e:
            print( e.args)
        """

        print('Checking for Nan rows...')
        nandf = self.raw_df[self.raw_df.isnull().any(axis=1)]
        if not nandf.empty:
            print(nandf)

        # For VS events, set direction code to X, since the direction is unknown
        #self.raw_df.Dir[self.raw_df[self.raw_df.TypeCode == 'VS'].index] = 'X'
        # drop rows with unwanted type codes (HEARTB)

        print('Pruning unwanted event types...')
        self.raw_df = self.raw_df.drop(self.raw_df[self.raw_df.EventCode == 'HEARTB'].index)
        self.raw_df = self.raw_df.drop(self.raw_df[self.raw_df.EventCode == 'VSX'].index)
        self.raw_df.reset_index()
        print('Total Records after pruning:', self.raw_df.shape)

        categ_features = pd.get_dummies(self.raw_df[['PeriodCode', 'EventDir', 'MarketTrend_D', 'MarketTrend_60', 'MarketTrend_15', 'MarketTrend_5', 'MarketTrend_1']], drop_first=False)
        self.data = pd.concat([self.raw_df.Type, categ_features], axis=1)
        #self.data = self.raw_df[['Type']]
        #self.data = np.array(self.raw_df.Type)
        #self.data['X'] = '{' + self.data['PeriodCode'] + ' ' + self.data['Dir'] + ' ' + self.data['TypeCode'] + '}'
        #labels = dftrim['Dir'] + '_' + dftrim['Period']
        self.labels = self.data.Type[1:]
        self.data = self.data[:-1]
        #all_data = pd.concat([data, labels], axis=0)
        #self.data.reset_index()
        self.nfeatures = self.data.shape[1]

        # scan for first row containing 'HIL*D' event code
        for idx in range(len(self.raw_df)):
            t = self.raw_df.Type.iloc[idx]
            mf = re.match(r'HILMF..D', t)
            ft = re.match(r'HILFT..D', t)
            if mf or ft:
                print('Found ', t, ' at index', idx)
                self.data=self.data[idx:]
                self.labels = self.labels[idx:]
                break


    def preprocess(self, vocab_file, tensor_file, encoding):

        #X = '[ ' + self.data.PeriodCode.astype(str) + ' ' + self.data.Dir + ' ' + self.data.TypeCode + ' ]'
        # save the data in a numpy file
        #self.tensor = np.array(self.data)
        #self.label_tensor = np.array(self.labels)
        #np.save(tensor_file, self.tensor)
        #self.vocab_size = len(self.featureCodes.eventCodeDict)

        self.vocab, self.words = self.build_vocab(self.data.Type)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.data['Type'] = np.array(list(map(self.vocab.get, self.data.Type)))
        self.tensor = np.array(self.data)
        self.label_tensor = np.array(list(map(self.vocab.get, self.labels)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)


    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / (self.batch_size * self.seq_length))
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size smaller."

        # truncate input tensor shape [n, self.nfeatures] to even number of full batches
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.label_tensor = self.label_tensor[:self.num_batches * self.batch_size * self.seq_length]

        self.x_batches = np.split(self.tensor.reshape((-1, self.seq_length, self.nfeatures)),
                                  self.num_batches, axis=0)
        self.y_batches = np.split(self.label_tensor.reshape(-1, self.seq_length),
                                  self.num_batches, axis=0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
