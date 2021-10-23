#
# common code given to the students
#


import requests
import urllib.parse
import urllib.request
import gensim


from collections import namedtuple
class Config(namedtuple('Config', ['doc', 'size', 'window', 'min_count', 'sg', 'negative', 'iter', 'name'])):

    def __str__(self):
        # skip doc
        fmt = "doc_len:{}, size:{}, window:{}, min_count:{}, sg:{}, negative:{}, iter:{}"
        return fmt.format(len(self.doc), self.size,
                          self.window, self.min_count,
                          self.sg, self.negative, self.iter)

def build_config(doc, size=10, window=5, min_count=5, sg=0, negative=3, iter=25, name=''):
  return Config(doc=doc, size=size, window=window, min_count=min_count, sg=sg, negative=negative, iter=iter, name=name)

def build_model(config):

    # export PYTHONHASHSEED=1
    model = gensim.models.Word2Vec(
        sentences=config.doc[0:3],   # each sentence is a HP book
        size=min(config.size, 300),  # how big the output vectors (spacy == 300)
        window=config.window,        # size of window around the target word
        min_count=config.min_count,  # ignore words that occur less than 2 times
        sg=config.sg,                # 0 == CBOW (default) 1 == skip gram

        negative=config.negative,

        # hs=1,
        # negative=0,
        # keep these the same
        workers=1,  # threads to use
        # sample=1e-3,
        iter=min(config.iter, 100),
    )
    # model.train(doc, total_examples=len(doc), epochs=100)
    return model

tests = [
    (['mcgonagall'],                [], ['professor'], ),          # 0
    (['ron'],                       [], ['hermione', 'harry'], ),  # 1
    (['seeker', 'quidditch'],       [], ['team'], ),               # 2
    (['harry', 'potter', 'school'], [], ['hogwarts'], ),           # 3
    (['gryffindor'],                [], ['hogwarts', 'house'], ),  # 4
    (['ron', 'hermione'],           [], ['harry'], ),              # 5
    (['ron', 'woman'],         ['man'], ['hermione'], ),           # 6
    (['hagrid'],                    [], ['dumbledore'], ),         # 7
    (['wizard'],                    [], ['witch'], ),              # 8
    (['weasley'],                   [], ['george', 'percy'], ),    # 9
    (['ravenclaw'],                 [], ['hufflepuff'], ),         # 10
    (['muggle', 'magic'],           [], ['wizard'], ),             # 11
    (['wizard'],     ['witch', 'dark'], ['potter'], ),             # 12
    (['house'],                     [], ['gryffindor'], ),         # 13
    (['house', 'evil'],             [], ['slytherin'], ),          # 14 
    (['voldemort'], ['evil'], ['dumbledore'],),                    # 15
    (['slytherin', 'good'], ['student'], ['malfoy', 'lucius'],),   # 16
    (['mcgonagall', 'good'], ['witch'], ['dumbledore']),           # 17
    (['harry', 'aunt'], [], ['petunia']),                          # 18
    (['you-know-who'], [], ['voldemort'])                          # 19
]
