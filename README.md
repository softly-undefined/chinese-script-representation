# Created 2/3/26 Eric Bennett

## Project Proposal

Large Language Models (LLMs) are a new form of AI increasingly being used around the world for a variety of tasks, including writing, education, and information retrieval (https://www.nber.org/system/files/working_papers/w34255/w34255.pdf). With this in mind, the world's language order is being condensed, and manipulated, through these tools. At the same time, there is evidence that humans are beginning to be influenced by LLM language in their own speech (https://arxiv.org/abs/2409.01754). The standards of multilinguality of these models are being set by the companies and organizations creating the models, primarily through competition between the US and China. LLMs are biased towards use of English and perform worse on non-English languages (https://arxiv.org/pdf/2003.11080), a predictable consequence of their English-centric internet training data. LLMs show differential performance between Traditional Chinese and Simplified Chinese, with a bias towards using Mainland-China specific terms (https://arxiv.org/pdf/2505.22645) and lesser performance on Traditional Chinese (https://arxiv.org/pdf/2403.01858), seen especially in Chinese-centric models.

With this in mind, I'm interested in approaching the following questions:
- Do LLMs treat Traditional and Simplified Chinese as distinct linguistic systems, or orthographic variants of a single language?
- Can we identify Chinese characters whose meanings are underspecified at the character level and only become stable once context is added using computational methods? Is there evidence Simplified Chinese is more reliant on context than Traditional, and if so what specific vocabulary are particularly susceptible?

To answer these questions, I plan to examine a dataset of chinese characters which in Simplified represent merges of 2 or more Traditional characters (https://en.wikipedia.org/wiki/Ambiguities_in_Chinese_character_simplification), as well as using a large dataset of multiple choice question/answer pairs in Traditional and Simplified Chinese (https://arxiv.org/pdf/2403.01858) alongside open source LLMs, and simpler computational language models like BERT.

## Assignment Description (Chin307)
The final project cultivates your ability to do research on topic of interest to you, critically employing existing scholarship and methodology. It is a three-step process. First, you will submit your topic with key references and discuss it with your professor. Second, you will present your research to the class, including your argument/hypothesis, methodology and scholarship, and your findings and conclusion. The last step is to submit your well-written paper formatted in the APA style.

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

0. Get a better understanding of the differences between Simplified and Traditional Chinese
    - Are there really only 190 examples of condensed words?
    - Rescrape the wikipedia to get all the disambiguated character data (might as well)
        - pull the mapping tables from opencc (how does opencc understand the conversion problem algorithmically)
        - ALSO get 1:1 mappings (characters that theoretically have the same meaning in zh-hant and zh-hans)

1. Collect an extensive dataset of characters being used in context.
    - Wikipedia dumps? (only collect the traditional one then convert with OpenCC)
    - Check the occurences of ambiguous words in the TMMLU+ dataset

1. View embedding representations of Simplified vs. Traditional (BERT models)
    - Model: mBERT (ensure both zh-hant and zh-hans are covered)
        1. TOKENIZATION: Check the tokenization process (ensure that overlap are considered simplified characters by the tokenization process)
        2. Static Embeddings: outside of a sentence (on their own)
            - Tells us: which characters' traditional senses are the farthest from the simplified ones (perhaps just on characters with multiple traditional characters merging into a brand new simplified character)
        3. Contextual Embeddings: in the context of a sentence
            - Tells us: same as static but now *IN CONTEXT* 
            - Extract from multiple layers (show semantic divergence between trad and simplified usage of identical characters through the layers?)
            - can reverse engineer this into a test of the model's disambiguation abilities, find which characters are easier or harder to disambiguate (how does opencc do this?)

2. Look at this behavior in more modern LLMs 
    - Models: Chinese-LLAMA, Taiwan-LLAMA, base-LLAMA, Chinese model(qwen?), English non-LLAMA model (Mistral?, )
    - Steps:
        1. TOKENIZATION: how does it work for each of these models?
        2. Contextual Embeddings
            - Sense Disambiguation
                - condition on the beginning of the sentence, observe the logits for traditional and simplified
                - when in all simplified does it consider traditional?

3. Extend this to downstream capabilities?
    - TMMLU+ dataset
        1. how many occurences are there of disambiguated characters? statistically significant?
            - If it is statistically significant than just use this instead of wikipedia dump
            - 
    - With this data I can now do:
        - create a sense disambiguation dataset? (surely this exists)

## Perhaps out of scope but interesting extensions

- words which sound different in different contexts? de vs. dei, etc.
- PINYIN: Wade Guilles system vs. Mainland Chinese PINYIN

random thought but IN DECODER MODELS THERE IS ONLY CAUSAL ATTENTION (only sees to the left)