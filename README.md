# Created 2/3/26 Eric Bennett

## File Structure

Data/..
- Contains the base data

## Project Proposal

Large Language Models (LLMs) are a new form of AI increasingly being used around the world for a variety of tasks, including writing, education, and information retrieval (https://www.semanticscholar.org/paper/How-People-Use-ChatGPT-Chatterji-Cunningham/7507725cd23d8c03e952bcb0196a17175b83c15a). With this in mind, the world's language order is being condensed, and manipulated, through these tools. At the same time, there is evidence that humans are beginning to be influenced by LLM language in their own speech (https://www.semanticscholar.org/paper/Empirical-evidence-of-Large-Language-Model's-on-Yakura-Lopez-Lopez/e030dc28ec35f4931b97622dab261134af744b81). The standards of multilinguality of these models are being set by the companies and organizations creating the models, primarily through competition between the US and China. LLMs are biased towards use of English and perform worse on non-English languages (https://arxiv.org/pdf/2003.11080), a predictable consequence of their English-centric internet training data. LLMs show differential performance between Traditional Chinese and Simplified Chinese, with a bias towards using Mainland-China specific terms (https://arxiv.org/pdf/2505.22645) and lesser performance on Traditional Chinese (https://arxiv.org/pdf/2403.01858), seen especially in Chinese-centric models.

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
    Total entries: 193
    Total unique characters across all forms: 438

    Entries by number of traditional counterparts:
    1 counterpart: 146
    2 counterparts: 42
    3 counterparts: 4
    4 counterparts: 1

    Entries where a traditional counterpart matches the simplified form:
    Count: 157
    Percentage: 81.35%

## The Plan

0. Get a better understanding of the differences between Simplified and Traditional Chinese
    - Are there really only 190 examples of condensed words?
        - Also looking at 1:1 mappings now
    - Rescrape the wikipedia to get all the disambiguated character data (might as well)
        - pull the mapping tables from opencc (how does opencc understand the conversion problem algorithmically)
        - ALSO get 1:1 mappings (characters that theoretically have the same meaning in zh-hant and zh-hans)

1. Collect an extensive dataset of characters being used in context.
    - Wikipedia dumps? (only collect the traditional one then convert with OpenCC) https://dumps.wikimedia.org/zhwiki/latest/
    - https://www.mediawiki.org/wiki/Writing_systems/LanguageConverter
    - Check the occurences of ambiguous words in the TMMLU+ dataset

1. View embedding representations of Simplified vs. Traditional (BERT models)
    - Model: mBERT (ensure both zh-hant and zh-hans are covered)
        1. TOKENIZATION: Check the tokenization process (ensure that overlap are considered simplified characters by the tokenization process)
            - Next steps:
                - Get a better list of characters (not just 2:1, but also 1:1!)
        2. Static Embeddings: outside of a sentence (on their own)
            - Tells us: which characters' traditional senses are the farthest from the simplified ones (perhaps just on characters with multiple traditional characters merging into a brand new simplified character)
            - Next steps:
                - Check for chars which can't be tokenized
                - Visualize the geometric distances between traditional and Simplified representations
                    - Visualize this throughout the neural network
                - Quantify and rank those differences using entropy between representations (difficult to do in 2:1, 1:2, 1:3, etc.)
        3. Contextual Embeddings: in the context of a sentence
            - Tells us: same as static but now *IN CONTEXT* 
            - Extract from multiple layers (show semantic divergence between trad and simplified usage of identical characters through the layers?)
            - can reverse engineer this into a test of the model's disambiguation abilities, find which characters are easier or harder to disambiguate (how does opencc do this?)
            - Next steps:
                - Clean the zh_hans and zh_hant stuff
                - Extract


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


read README.md,

sketch out what you think should be done with the mBERT-experiments directory and I will give feedback


## Update 2/5/26

These are items that have multiple Simplified substitutions for a single Traditional:
  // { "simplified": "线", "traditional": ["線"] },
  // { "simplified": "缐", "traditional": ["線"] },
  // { "simplified": "著", "traditional": ["著"] },
  // { "simplified": "着", "traditional": ["著"] }

I'm having trouble with finding a good massive corpus of Traditional Chinese text. Chinese wikipedia contains both, but worried it isn't truly traditional text. It's complex because the culture is so important to this.

NOTE: i'm using opencc to convert to traditional and simplified, acknowledge this.

TODO:
- switch split.py to use opencc to confirm simplified vs traditional

## Update 2/7/26

I have mBERT up and running (very lightweight on mBERT). Now it's time to start going for it! Need to do this in a very structured way, and make the code for static embeddings so that it easily is usable for the contextual embeddings.

UPDATE: Starting 2:1 testing. Very difficult. Not sure what to think of it yet!
interesting plot but hard to make sense of