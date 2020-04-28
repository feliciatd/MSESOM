import re

def str_wrap_chem(word):

    # replace space with linebreak
    word = word.replace(' ', '\n')

    # if find cyclo, fluoro, bromo, fluoro, bromo, or hydro,
    #     linebreak after that
    pattern = r'cyclo|chloro|bromo|fluoro|hydro'
    match = re.search(pattern, word, re.IGNORECASE)
    if match:
        word = word[:match.start()] + word[match.start():match.end()] \
            + '-\n' + word[match.end():]

    # if find methyl with any subsequent letter, linebreak after that
    pattern = r'methyl.'
    match = re.search(pattern, word, re.IGNORECASE)
    if match:
        word = word[:match.start()] + word[match.start():match.end()-1] \
            + '-\n' + word[match.end()-1:]

    return word
