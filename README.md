# Created 2/3/26 Eric Bennett

## The Topic
The final project cultivates your ability to do research on topic of interest to you, critically employing existing scholarship and methodology. It is a three-step process. First, you will submit your topic with key references and discuss it with your professor. Second, you will present your research to the class, including your argument/hypothesis, methodology and scholarship, and your findings and conclusion. The last step is to submit your well-written paper formatted in the APA style.

## The Idea
Observe the differences between Traditional and Simplified Chinese in Large Language Models

## The Data
I found a list on Wikipedia of words where more than 1 Traditional Character is compressed into a single Simplified Character [here](https://en.wikipedia.org/wiki/Ambiguities_in_Chinese_character_simplification)

Also seems like a good resource [here](https://pages.ucsd.edu/~dkjordan/chin/SimplifiedCharacters.html#prin4)

Maybe this for more data? [here](https://zh.wikipedia.org/zh-hant/簡繁轉換一對多列表)

Description of the data (created by first_check.py):
    Total entries: 190
    Total unique characters across all forms: 388

    Entries by number of traditional counterparts:
    1 counterpart: 146
    2 counterparts: 43
    3 counterparts: 1

    Entries where a traditional counterpart matches the simplified form:
    Count: 154
    Percentage: 81.05%

## The Plan

1. View embedding representations of Simplified vs. Traditional
2. Look in context of sentences (is the simplified variant represented differently in context of a traditional sentence)


## What I need
Example sentences (preferably multiple) in both traditional and simplified characters
- What dataset could I pull these from? Generate synthetically?

## Thoughts

add words which sound different in different contexts? de vs. dei, etc.

look at PINYIN / wade guilles