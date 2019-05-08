"""
note that preprocess for nl shuold only in method label2tokens(), not in any other places.
also, preprocess for label shuold only be used in method generate_random_walks(), not other places.
"""
import nltk
import re
import collections
import enchant
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# from build_data.nl_transform import nl_transformer
class preprocessor:
    def __init__(self):
        self.info = 'nltk tokenizer'

    @staticmethod
    def tokenize(input):
        """
        handle following  tokenizing cases:
        1. split by ' '
        2. split by '.'
        3. split by '$'
        4. split by '_' (edge type data_flow)
        5. clean
        6. camel style names
        """
        splited = []
        for t1 in input.split(' '):
            for t2 in t1.split('.'):
                for t3 in t2.split('$'):
                    for t4 in t3.split('_'):
                        if t4 != None and t4 != '':
                            splited.append(t4)

        splited = preprocessor.clean_nl_tokens(splited)

        underlines = []
        for t in splited:
            t_u = ''
            for _s_ in t:
                if isinstance(_s_, str):
                    t_u += _s_ if _s_.islower() else '_' + _s_.lower()
            underlines.append(t_u)

        camel_splited = []
        for tu in underlines:
            for t in tu.split('_'):
                if t != None and t != '':
                    camel_splited.append(t)

        '''
        deprecated for it may cause un wanted split
        '''
        # latent_splited = []
        # d = enchant.Dict('en_US')
        # latent_words = []
        # for t in camel_splited:
        #     word = ''
        #     buffer = collections.deque(maxlen=len(t))
        #     for _s_ in t:
        #         buffer.append(_s_)
        #
        #     for i in range(len(buffer)):
        #         word += buffer.popleft()
        #         if d.check(word) and len(word) > 1:
        #             print(word)
        #             latent_words.append(word)
        #             word = ''

        return camel_splited


    @staticmethod
    def remove_stop_words(tokens):
        stop_list = [':', '@', '=', '(', ')', '()', '\'\'', '( )', ',', '.', '$', '!']
        stopWords = set(stopwords.words('english'))
        stop_list += stopWords
        filtered = [t for t in tokens if (t not in stop_list)]
        filtered = [t for t in filtered if len(t) != 1]
        return filtered

    @staticmethod
    def clean_nl_tokens(tokens):
        """
        target: single token, not suitable for a complete nl sentence or a jimple representation
        for the ops maybe hurt the origin structure of sentence / jimple
        :param tokens:
        :return:
        """
        pattern1 = re.compile(r'[^a-zA-Z0-9]')
        move_targets = [str(i) for i in range(10)]
        move_targets = tuple(move_targets)
        res = []
        for t in tokens:
            t_ = re.sub(pattern1, '', t)
            if t_.startswith(move_targets):
                t_ = t_[1:]
            if t_.endswith(move_targets):
                t_ = t_[:-1]
            if t_ != None and t_ != '':
                res.append(t_)
            else:
                print('clean exception')
                print(t, t_)
        return res

    @staticmethod
    def lemmatize(tokens):
        res = []
        lemma = WordNetLemmatizer()
        pos_sent = nltk.pos_tag(tokens)
        for pos in pos_sent:
            tag = preprocessor.get_wordnet_pos(pos[1]) or 'v'
            word = pos[0]
            # print(word,tag)
            t = lemma.lemmatize(word, pos='v')
            res.append(t)
        return res

    @staticmethod
    def stemm(tokens):
        res = []
        stemmer = PorterStemmer()
        for t in tokens:
            if t.endswith('s'):
                t = stemmer.stem(t)
            res. append(t)
        return res

    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def clean_label(jimple):
        res = jimple.strip('\n')
        res = res.strip('\"')
        res = res.strip('\'')
        return res

    @staticmethod
    def if_in_glove(target):
        glove_words = []
        glove_file = '/home/qwe/zfy_data/SoC/data_new/nl_pre/glove/glove.6B.100d.txt'
        f = open(glove_file, 'r')
        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            glove_words.append(word)

        if target in glove_words and len(target) > 1:
            return True
        else:
            return False

    @staticmethod
    def if_in_wordnet(target):
        d = enchant.Dict('en_US')
        if d.check(target) and len(target) > 1:
            return True
        else:
            return False

    @staticmethod
    def if_in_heuristic(target):
        path = '/home/qwe/zfy_data/SoC/data_new/nl_pre/glove/outlaw.txt'
        f = open(path, 'r')
        all = []
        for line in f.readlines():
            w = line.strip('\n')
            all.append(w)
        all = set(all)
        if target in all:
            return True
        else:
            return False


    @staticmethod
    def make_glove(tokens):
        res = []
        for t in tokens:
            if not preprocessor.if_in_glove(t):
                if preprocessor.if_in_glove(t[:-1]):
                    res.append(t[:-1])
                elif preprocessor.if_in_glove(t + 's'):
                    res.append(t + 's')
                elif preprocessor.if_in_glove(t + 'es'):
                    res.append(t + 'es')
                else:
                    target = ''
                    buffer = collections.deque(maxlen=len(t))
                    for _s_ in t:
                        buffer.append(_s_)
                    while(len(buffer) > 0):
                        target += buffer.popleft()
                        if preprocessor.if_in_heuristic(target):
                            res.append(target)
                            target = ''
            else:
                res.append(t)
        return res



