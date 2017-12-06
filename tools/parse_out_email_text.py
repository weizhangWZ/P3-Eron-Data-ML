#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
error = [""]

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words = stemize(text_string)

    return words


def stemize(text_string):
    new_string,initial = "",0
    words_original = text_string.split()
    #print string
    stemmer = SnowballStemmer("english")
    for word in words_original:
        if initial == 0:
            initial = 1
            new_string = stemmer.stem(word)
        elif word not in error:
            new_string = new_string + " " + stemmer.stem(word)
    return new_string

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

