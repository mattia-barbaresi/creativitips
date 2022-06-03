# frokm: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
# see also: https://github.com/shubhamjn1/TextBlob

# Stop words
# importing stop words from English language.
import spacy
# Word tokenization
from spacy.lang.en import English
# importing the model en_core_web_sm of English for vocabluary, syntax & entities
import en_core_web_sm

from spacy import displacy

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """When learning data science, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""
text2 = "KYŌGEN: I am someone who lives in the Kitayama area of the capital." \
        " I have heard that near here, at Unrin-in, there will be memorial services for the flowers offered over the summer. " \
        "This being the case, I too, intend to go there. Truly, as few people are going there this evening, it is rather melancholy. " \
        "WAKI: Ho, there! Are you someone from around here?" \
        "KYŌGEN: Indeed, I am a person from this area." \
        "WAKI: If that’s the case, first of all, could you come closer? There is something I would like to ask you." \
        "KYŌGEN: Certainly. Well then, what is the matter you would like to ask me about?" \
        "WAKI: While it isn’t a circumstance of much import I would ask of you, I would like you to tell me the particulars of the story of Lady Yūgao of the Evening Faces, who lived here long ago." \
        "KYŌGEN: This is an unexpected request you have made of me. I don’t know about the situation in much detail, but I will do the best I can to tell you the story as I have heard it. Well then, the one known as Lady Yūgao was the daughter of a middle captain of the third rank, and owing to certain circumstances, she came to live in seclusion in the area of Gojō. Also, at that time Hikaru Genji, was paying regular visits to the home of Lady Rokujo. On occasion, he would pass through the Gojō area and it happened that there was a simple dwelling there, with ‘evening faces’ blooming in splendid profusion. He called Koremitsu to him, and requested that he go plucked a blossom. When Koremitsu spoke to his attendant, expressing this desire, a child came out of the house, and having asked that he wait there for a while, soon after offered him a white fan, its edges perfumed with incense, on which rested a blossom of the 'evening faces.' The attendant took this and returned with it to Koremitsu. Koremitsu went with it to Genji, who, seeing then that there was a poem on the edge of the fan, quickly wrote a verse in reply. Because of that flower, he was drawn to the Lady Yūgao, of the 'evening faces,' and as the vows they shared were not shallow, after that he came to stay together with her there at her dwelling. Even as the purity of the many pledges they exchanged were profound, by the Sacred Treasures! - Mysteriously, Lady Yūgao was possessed by some sort of vengeful spirit, and it is said that all was lost. As to Kikaru Genji’s grief, its depth was limitless, as there was nothing to be done for her, it is said. In any event, those are the details, at least in broad terms, as I know them. But why are you interested in this and asking me about this matter now? I find it suspicious." \
        "WAKI: I’m grateful that you described the matter so thoroughly. When I was performing a memorial service for the flowers offered during the summer, a woman appeared and gave me a white flower. I asked who, and what sort of person, she was with the flower, to which she replied speaking of the Lady of the Evening Faces as though she spoke for herself. Then, she disappeared in the shadow of the flower. That’s what happened." \
        "KYŌGEN: How astonishing! What a wondrous thing you have related to me. Now then, if I may say what I suppose, it can only be that when you were performing the memorial service for flowers, the spirit of the “yūgao flower”, and also the ghost of Lady Yūgao, Lady of the Evening Faces, of long ago appeared before you. It can only be that. This being the case, I believe that due to the merit emanating from your Reverence, the spirit of Lady Yūgao appeared to you in order to ask that you offer prayers for her deliverance. If that is the case, you should go now to the Gojō area, where a miracle will surely occur. Then, you should pray for her enlightenment, as she requested of you." \
        "WAKI: As I also believe this to be true, I will go to the Gojō area and pray there for the ghost of Lady Yūgao and her deliverance." \
        "KYŌGEN: As you will be going there shortly, I’ll follow after your Reverence in a while." \
        "WAKI: In due time then, I will expect you there." \
        "KYŌGEN: I’ll do as you say."

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text2)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
print(token_list)

# sentence tokenization

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Add the 'sentencizer' component to the pipeline
nlp.add_pipe('sentencizer')

text = """When learning data science, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text2)

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
print(sents_list)

# Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))

# Printing first ten stop words:
print('First ten stop words: %s' % list(spacy_stopwords)[:20])

# Implementation of stop words:
filtered_sent = []

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text2)

# filtering stop words
for word in doc:
    if word.is_stop == False:
        filtered_sent.append(word)
print("Filtered Sentence:", filtered_sent)

# Implementing lemmatization
lem = nlp("run runs running runner")
# finding lemma for each word
for word in lem:
    print(word.text, word.lemma_)

# POS tagging


# load en_core_web_sm of English for vocabluary, syntax & entities
nlp = en_core_web_sm.load()

#  "nlp" Objectis used to create documents with linguistic annotations.
docs = nlp(u"All is well that ends well.")

for word in docs:
    print(word.text, word.pos_)

# for visualization of Entity detection importing displacy from spacy:


nytimes = nlp(u"""New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases.

At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday.

The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.""")

entities = [(i, i.label_, i.label) for i in nytimes.ents]
print("entities:",entities)

displacy.render(nytimes, style="ent", jupyter=True)

docp = nlp(" In pursuit of a wall, President Trump ran into one.")

for chunk in docp.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

# import en_core_web_sm
# nlp = en_core_web_sm.load()
# mango = nlp(u'mango')
# print(mango.vector.shape)
# print(mango.vector)
