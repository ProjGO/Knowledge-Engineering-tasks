import nltk
newfile = open('E:/Study/第四学期/知识工程/task2/4911904conll-corpora/conll-corpora/CoNLL-2003/eng.testa.txt')
text = newfile.read()
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
a1 = str(entities)
print(entities)
'''import nltk
nltk.download('words')'''