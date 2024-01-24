# Table of Contents
1. [Introduction](#introduction)
2. [Attention Model](#attention-model)
3. [How Attention Works?](#how-attention-works)
4. [Code Walk-through](#code-walk-through)
   1. [Self Attention](#self-attention)
   2. [Multi Head Attention](#multi-head-attention)
   3. [Comparison between Self and Multi Head Attention Mechanism](#comparison-between-self-and-multi-head-attention-mechanism)
5. [Conclusion](#conclusion)


## 1. Introduction <a name="introduction"></a>
Attention mechanism is one of the recent advancements in Deep learning especially for Natural language processing tasks like Machine translation, Image Captioning, dialogue generation etc. It is a mechanism that is developed to increase the performance of encoder decoder(seq2seq) RNN model. In this blog post I will try to explain the attention mechanism for the text classification task.

## 2. Attention Model <a name="attention-model"></a>
Attention is proposed as a solution to the limitation of the Encoder-Decoder model which encodes the input sequence to one fixed length vector from which to decode the output at each time step. This issue is believed to a problem when decoding long sequences because it make difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus.

```Attention is proposed to be a method of align and translate.```

In attention when the model is trying to predict the next word it searches for a set of positions in a source sentence where the most relevant information is concentrated. The model then predicts next word based on context vectors associated with these source positions and all the previous generated target words.

Instead of encoding the input sequence into a single fixed context vector, the attention model develops a context vector that is filtered specifically for each output time step.

## 3. How Attention Works? <a name="how-attention-works"></a>
The basic idea in Attention is that each time the model tries to predict an output word, it only uses parts of an input where the most relevant information is concentrated instead of an entire sentence i.e it tries to give more importance to the few input words. Let’s see how it works:

![attention](https://miro.medium.com/v2/resize:fit:828/format:webp/1*wa4zt-LcMWRIYLfiHfBKvA.png "Attention Mechanism Illustration")

In attention, the encoder works as similar to encoder-decoder model but the decoder behaves differently. As you can see from a picture, the decoder’s hidden state is computed with a context vector, the previous output and the previous hidden state and also it has separate context vector c_i for each target word. These context vectors are computed as a weighted sum of activation states in forward and backward directions and alphas and these alphas denote how much attention is given by the input for the generation of output word.

![image](https://miro.medium.com/v2/resize:fit:456/format:webp/1*rjIVbJMcDi1ZMyZbblaunA.png "Context vector for output word 1")

Here, ‘a’ denotes the activation in backward and forward direction and alpha denotes the attention each input word gives to the output word.

## 4. Code Walk-through <a name="code-walk-through"></a>


### 4.1 Self Attention <a name="self-attention"></a>
- Utilizes a self-attention layer and bi-directional RNN for embedding.
- Random initialization is used for embedding.
- Last layer is densely connected for binary classification.

### 4.2 Multi Head Attention <a name="multi-head-attention"></a>
- Employs a Multi Head Attention layer with LSTM units for embedding.
- Similar to self-attention, random initialization is used.
- Last layer is densely connected for binary classification.

### 4.3 Comparison between Self and Multi Head Attention Mechanism
Let's delve deeper into the differences between multi-head attention and self-attention (also known as single-head attention) mechanisms:

1. **Self-Attention (Single-Head Attention):**
   - **Definition:** In self-attention, also known as intra-attention, the model considers the relationships within a sequence (e.g., a sentence) by allowing each element to focus on other elements in the same sequence.
   - **Operation:** For each element in the sequence, self-attention computes attention scores with all other elements. These scores determine the importance or relevance of each element to the current one. The final output is a weighted sum of all elements, creating a context vector for each element.

2. **Multi-Head Attention:**
   - **Definition:** Multi-head attention extends the self-attention mechanism by employing multiple sets (or "heads") of attention weights simultaneously. Each head processes the input sequence independently and contributes to the final output.
   - **Operation:** The input sequence is transformed into multiple representations, one for each attention head. Each head has its own set of attention weights, allowing it to capture different aspects or patterns within the sequence. The outputs from all heads are concatenated and linearly transformed to produce the final output.

**Key Differences:**

- **Parallelization and Diversification:**
  - **Self-Attention:** Operates with a single set of attention weights, capturing relationships within the sequence.
  - **Multi-Head Attention:** Utilizes multiple sets of attention weights in parallel, capturing diverse patterns and relationships simultaneously.

- **Expressiveness and Complexity:**
  - **Self-Attention:** Captures dependencies within the sequence but may have limitations in representing complex relationships.
  - **Multi-Head Attention:** Allows the model to capture richer and more nuanced relationships by employing multiple attention mechanisms.

- **Parameterization:**
  - **Self-Attention:** Has fewer parameters compared to multi-head attention since it uses a single set of attention weights.
  - **Multi-Head Attention:** Involves more parameters due to the use of multiple sets of attention weights, potentially increasing model expressiveness.

- **Performance:**
  - **Self-Attention:** Can work well for capturing local dependencies within a sequence.
  - **Multi-Head Attention:** Excels in capturing both local and global dependencies, providing a more powerful mechanism for learning complex relationships.

In summary, while self-attention focuses on relationships within a sequence using a single set of attention weights, multi-head attention enhances this by leveraging multiple sets of attention weights in parallel, allowing the model to capture a broader range of patterns and dependencies. The introduction of multiple heads increases the model's capacity to learn complex relationships within the data.

## 5. Conclusion <a name="conclusion"></a>
The Attention mechanism proves to be valuable in NLP tasks, enhancing accuracy and BLEU score. However, it comes with the drawback of being time-consuming and challenging to parallelize.



