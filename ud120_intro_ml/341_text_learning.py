import os
import string
import pickle

from nltk.stem.snowball import SnowballStemmer

def parse_out_text(fh):
    print('======== parse out text ')
    fh.seek(0)
    all_text = fh.read()
    content = all_text.split('X-FileName:')
    words = ''

    stemmer = SnowballStemmer('english')
    if len(content) > 1:
        text_string = content[1].translate(str.maketrans('', '', string.punctuation))
        text_words = text_string.split()
        words = [stemmer.stem(word) for word in text_words]
        words = ' '.join(words)
        print(words)
    return words

from_sara = open('data/from_sara.txt', 'r')
from_chris = open('data/from_chris.txt', 'r')

from_data = []
word_data = []

temp_counter = 0
word_to_be_replaced = ['sara', 'shackleton', 'chris', 'germani']

for name, from_person in [('sara', from_sara), ('chris', from_chris)]:
    for path in from_person:
        temp_counter += 1
        if temp_counter < 3:
            path = os.path.join('194_enron', path[:-1])
            print(path)
            if os.path.exists(path):
                email = open(path, 'r')
                email_content = parse_out_text(email)
                final_email_content = ''
                for word in email_content.split():
                    if word not in word_to_be_replaced:
                        final_email_content += '{} '.format(word)
                final_email_content = final_email_content.rstrip()
                print('======== email content after replacement')
                print(final_email_content)
                word_data.append(final_email_content)
                print(name)
                if name == 'sara':
                    from_data.append(0)
                elif name == 'chris':
                    from_data.append(1)
                
                email.close()
print('emails processed')
print(from_data)
print(word_data)

from_sara.close()
from_chris.close()

pickle.dump(word_data, open('data/your_word_data.pkl', 'wb'))
pickle.dump(from_data, open('data/your_email_authors.pkl', 'wb'))