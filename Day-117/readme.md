# Named Entity Recongition using Natural language processing

Named Entity Recognition (NER) is a technique in natural language processing (NLP) that focuses on identifying and classifying entities. The purpose of NER is to automatically extract structured information from unstructured text, enabling machines to understand and categorize entities in a meaningful manner for various applications like text summarization, building knowledge graphs, question answering, and knowledge graph construction. The article explores the fundamentals, methods and implementation of the NER model.

### What is Named Entity Recognition (NER)?
Name-entity recognition (NER) is also referred to as entity identification, entity chunking, and entity extraction. NER is the component of information extraction that aims to identify and categorize named entities within unstructured text. NER involves the identification of key information in the text and classification into a set of predefined categories. An entity is the thing that is consistently talked about or refer to in the text, such as person names, organizations, locations, time expressions, quantities, percentages and more predefined categories.

NER system fin applications across various domains, including question answering, information retrieval and machine translation. NER plays an important role in enhancing the precision of other NLP tasks like part-of-speech tagging and parsing. At its core, NLP is just a two-step process, below are the two steps that are involved:

- Detecting the entities from the text
- Classifying them into different categories

**Ambiguity in NER**
For a person, the category definition is intuitively quite clear, but for computers, there is some ambiguity in classification. Let’s look at some ambiguous examples:

- England (Organization) won the 2019 world cup vs The 2019 world cup happened in England (Location).
- Washington (Location) is the capital of the US vs The first president of the US was Washington (Person).

**How Named Entity Recognition (NER) works?**

- The working of Named Entity Recognition is discussed below:

- The NER system analyses the entire input text to identify and locate the named entities.

- The system then identifies the sentence boundaries by considering capitalization rules. It recognizes the end of the sentence when a word starts with a capital letter, assuming it could be the beginning of a new sentence. Knowing sentence boundaries aids in contextualizing entities within the text, allowing the model to understand relationships and meanings.

- NER can be trained to classify entire documents into different types, such as invoices, receipts, or passports. Document classification enhances the versatility of NER, allowing it to adapt its entity recognition based on the specific characteristics and context of different document types.

- NER employs machine learning algorithms, including supervised learning, to analyze labeled datasets. These datasets contain examples of annotated entities, guiding the model in recognizing similar entities in new, unseen data.

- Through multiple training iterations, the model refines its understanding of contextual features, syntactic structures, and entity patterns, continuously improving its accuracy over time.

- The model’s ability to adapt to new data allows it to handle variations in language, context, and entity types, making it more robust and effective.

### **Named Entity Recognition (NER) Methods**

**Lexicon Based Method**
The NER uses a dictionary with a list of words or terms. The process involves checking if any of these words are present in a given text. However, this approach isn’t comm**only used because it requires constant updating and careful maintenance of the dictionary to stay accurate and effective.

**Rule Based Method**
The Rule Based NER method uses a set of predefined rules guides the extraction of information. These rules are based on patterns and context. Pattern-based rules focus on the structure and form of words, looking at their morphological patterns. On the other hand, context-based rules consider the surrounding words or the context in which a word appears within the text document. This combination of pattern-based and context-based rules enhances the precision of information extraction in Named Entity Recognition (NER).

### Machine Learning-Based Method

**Multi-Class Classification with Machine Learning Algorithms**
- One way is to train the model for multi-class classification using different machine learning algorithms, but it requires a lot of labelling. In addition to labelling the model also requires a deep understanding of context to deal with the ambiguity of the sentences. This makes it a challenging task for a simple machine learning algorithm.

**Conditional Random Field (CRF)**

- Conditional random field is implemented by both NLP Speech Tagger and NLTK.  It is a probabilistic model that can be used to model sequential data such as words.

- The CRF can capture a deep understanding of the context of the sentence. In this model, the input 
 <div style="background-color:white;">

![image](https://quicklatex.com/cache3/bd/ql_b1ef4df3296150c875128fd1a962acbd_l3.svg)

![image](https://quicklatex.com/cache3/cb/ql_0ac5b7001bb7c22264fbd072657dc3cb_l3.svg )
</div>

**Deep Learning Based Method**

- Deep learning NER system is much more accurate than previous method, as it is capable to assemble words. This is due to the fact that it used a method called word embedding, that is capable of understanding the semantic and syntactic relationship between various words.

- It is also able to learn analyzes topic specific as well as high level words automatically.

- This makes deep learning NER applicable for performing multiple tasks. Deep learning can do most of the repetitive work itself, hence researchers for example can use their time more efficiently.

