# Probabilistic Latent Semantic Analysis (PLSA) Implementation

This repository contains an implementation of the Probabilistic Latent Semantic Analysis (PLSA) algorithm, a statistical technique used in natural language processing to uncover the underlying topics in a collection of documents.

## Features

- **Corpus Building:** Reads and processes a corpus of documents, handling both labeled and unlabeled data.
- **Vocabulary Construction:** Builds a vocabulary from the corpus while filtering out common stopwords.
- **Term-Document Matrix:** Constructs a term-document matrix to represent word counts in each document.
- **Initialization Methods:** Supports both random and uniform initialization of document-topic and topic-word probability distributions.
- **Expectation-Maximization (EM) Algorithm:** Iteratively refines the probability distributions using the EM algorithm to maximize the likelihood of the observed data.
- **Likelihood Calculation:** Calculates and tracks the log-likelihood of the model to monitor convergence.
- **Topic Analysis:** Outputs the top words associated with each discovered topic for easy interpretation.

## Probabilistic Latent Semantic Analysis (PLSA)

PLSA is a statistical technique used to analyze large collections of documents to uncover the hidden thematic structure within them. It operates under the assumption that documents are mixtures of topics, and topics are mixtures of words.

### Key Concepts:

- **Document-Topic Distribution (P(z|d)):** Each document is represented as a probability distribution over topics.
- **Topic-Word Distribution (P(w|z)):** Each topic is represented as a probability distribution over words.
- **Expectation-Maximization (EM) Algorithm:** The PLSA model is trained using the EM algorithm, which iteratively refines the distributions to maximize the likelihood of the observed data.

### Steps in PLSA:

1. **Initialization:** Start with initial estimates for the document-topic and topic-word distributions.
2. **E-Step (Expectation):** Calculate the probability distribution of topics given the words in each document.
3. **M-Step (Maximization):** Update the document-topic and topic-word distributions to better fit the observed data.
4. **Convergence:** Repeat the E and M steps until the likelihood of the data stabilizes.

PLSA helps in identifying the main themes in a document collection, making it useful for tasks like topic modeling, information retrieval, and text classification.

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your corpus file (e.g., `DBLP_1000.txt`) and place it in the project directory. The file should contain one document per line.

2. Run the main script:

   ```bash
   python plsa.py
   ```

3. The script will output the vocabulary size, the number of documents, and the top words for each topic after the EM algorithm converges.

## Results

Here are the top 10 words for each of the 4 topics discovered in a sample run, along with their inferred labels and explanations:

- **Topic 1: Performance and Algorithms**

  - Top 10 words: ['time', 'performance', 'memory', 'algorithm', 'cache', 'based', 'system', 'data', 'paper', 'using']
  - **Explanation:** This topic includes terms related to the performance of algorithms and systems, focusing on aspects like time, memory usage, and caching. It makes sense that these words are grouped together as they often appear in discussions about optimizing algorithm performance and system efficiency.

- **Topic 2: Software and Information Systems**

  - Top 10 words: ['data', 'systems', 'based', 'software', 'paper', 'system', 'information', 'model', 'user', 'approach']
  - **Explanation:** This topic is centered around software systems and information management. Terms like 'data', 'systems', 'software', and 'information' are commonly found together in papers discussing various approaches and models for handling and processing data in software systems.

- **Topic 3: Networks and Mobile Systems**

  - Top 10 words: ['network', 'based', 'nodes', 'networks', 'performance', 'paper', 'data', 'mobile', 'system', 'sensor']
  - **Explanation:** This topic focuses on network systems and mobile technologies. Words such as 'network', 'nodes', 'mobile', and 'sensor' indicate discussions related to the performance and architecture of networked and mobile systems, including the analysis of data within these contexts.

- **Topic 4: Data Analysis and Algorithms**
  - Top 10 words: ['data', 'based', 'method', 'results', 'paper', 'problem', 'algorithms', 'approach', 'algorithm', 'using']
  - **Explanation:** This topic revolves around data analysis methods and algorithms. Terms like 'data', 'method', 'results', and 'algorithms' suggest a focus on different approaches and techniques for analyzing data, solving problems, and reporting results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please contact [your email].
