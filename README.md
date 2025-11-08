# Understanding and Applying Text Embeddings

This repository contains personal notes, projects, and explorations based on the course **[Understanding and Applying Text Embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/)** by **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[Google Cloud](https://cloud.google.com/)**.

This course provides a comprehensive, hands-on introduction to the world of text embeddings, which are dense numerical representations of text that capture semantic meaning. These vectors are the foundational technology behind many of the AI applications we use daily, from Google Search and e-commerce recommendations to sophisticated question-answering systems.

The course leverages the **Vertex AI Text-Embeddings API** to generate these powerful representations. Through a series of practical labs and conceptual lectures, you will learn to apply text embeddings to a variety of core machine learning tasks, including:

  * **Classification:** Building models that can categorize text based on its semantic meaning.
  * **Outlier Detection:** Identifying semantically "different" or anomalous pieces of text from a large corpus.
  * **Text Clustering:** Automatically grouping unlabeled documents by their underlying topics or themes.
  * **Semantic Search:** Building search engines that find results based on *meaning* and *intent*, not just keyword matching.

A key part of the course focuses on integrating these components into end-to-end systems. You will explore how to fine-tune the behavior of Large Language Models (LLMs) by adjusting parameters like **temperature**, **top-k**, and **top-p**. You will also learn to use the open-source **ScaNN (Scalable Nearest Neighbors)** library for building highly efficient, large-scale vector search systems.

The capstone of the course involves combining these technologies—embeddings, vector search, and LLM generation—to build a powerful **Retrieval-Augmented Generation (RAG)** system, which can answer questions by referencing a private knowledge base, thereby grounding the LLM's responses in factual data.

## Course Topics

<details>
<summary><strong>1. Getting Started With Text Embeddings</strong></summary>

This initial module serves as a foundational introduction to the "why" and "what" of text embeddings. It bridges the gap between traditional natural language processing (NLP) techniques and the modern, vector-based approaches that power today's AI. It sets the stage by explaining the fundamental problem: computers do not understand text; they understand numbers. This module is about the journey of converting abstract language into a structured, mathematical format that a machine can process.

**The Problem with Text: From Words to Numbers**

The module begins by exploring *why* text is so difficult for computers. Language is inherently unstructured, categorical, and high-dimensional. A single word, "bank," can mean a financial institution or the side of a river. To a computer, the words "king" and "queen" are just as different as "king" and "apple"—they are simply distinct strings of characters.

**Traditional (Sparse) Representations**

To solve this, early NLP techniques focused on "sparse" vector representations:

  * **One-Hot Encoding:** This is the most basic method. If you have a vocabulary of 50,000 words, each word is represented by a 50,000-dimensional vector that is all zeros *except* for a single '1' at the index corresponding to that word.

      * `king` = `[0, 0, ..., 1, ..., 0, 0]` (at index 8012)
      * `queen` = `[0, 1, ..., 0, ..., 0, 0]` (at index 2045)
      * **Problems:**
        1.  **Massive Dimensionality:** The vectors are huge and unwieldy.
        2.  **Sparsity:** The vectors are almost entirely empty, making them computationally inefficient.
        3.  **No Semantic Meaning:** The "distance" (e.g., dot product) between any two one-hot vectors is always the same (zero). The model cannot learn that "king" is more related to "queen" than it is to "apple."

  * **Bag-of-Words (BoW):** This improves on one-hot encoding by representing an entire *document* as a single vector. The vector's length is still the vocabulary size, but each entry now holds the *count* of how many times a word appeared in the document.

      * `"The king and the queen"` -> `[... "and": 1, ... "king": 1, ... "queen": 1, ... "the": 2, ...]`
      * **Improvement:** Captures the topic or content of a document.
      * **Problems:** Still sparse, still massive, and crucially, it *loses all word order*. `"Man bites dog"` and `"Dog bites man"` would have identical BoW representations.

  * **TF-IDF (Term Frequency-Inverse Document Frequency):** This is a refinement of BoW. Instead of raw counts, it calculates a *score* for each word.

      * **Term Frequency (TF):** How often a word appears in *this* document. (Words used often are important).
      * **Inverse Document Frequency (IDF):** How rare a word is across *all* documents. (Words like "the" or "and" appear everywhere and are thus *not* informative, so they get a low score).
      * **TF-IDF Score:** `TF * IDF`. This score is high for words that are *frequent in this document* but *rare in the overall corpus*.
      * **Improvement:** This is a very strong baseline for tasks like information retrieval and document classification. It surfaces a document's key themes.
      * **Problems:** Still sparse, still has no concept of semantic similarity. "Awesome" and "incredible" are treated as two completely distinct, unrelated dimensions.

**The Embedding Revolution: Dense Representations**

This is where "embeddings" are introduced as the solution. An embedding is a **dense**, **low-dimensional** vector.

  * **Dense:** Unlike a 50,000-dimensional sparse vector, an embedding might only have 300, 512, or 768 dimensions. *Every* one of those dimensions has a non-zero value, a floating-point number.
  * **Low-Dimensional:** 768 dimensions is far more manageable than 50,000.
  * **Learned Representation:** This is the most important part. These vectors are not just *assigned*; they are *learned* by a machine learning model.

The model learns these vectors by processing billions of sentences, and it does so based on the **Distributional Hypothesis**: "A word is characterized by the company it keeps."

The model learns that words like "king," "queen," "prince," and "princess" frequently appear in similar contexts (e.g., near words like "throne," "palace," "royalty"). As a result, the model adjusts the 300 numbers in their respective vectors to be *close to each other* in this high-dimensional "vector space."

**The Power of Vector-Space Arithmetic**

The true "magic" of embeddings, which this module introduces, is that the *relationships* between words are captured in the geometry of this vector space. The famous canonical example is:

`vector("king") - vector("man") + vector("woman") ≈ vector("queen")`

This simple arithmetic demonstrates that the model has learned the concept of "gender" or "royalty" as *directions* or *axes* within this 300-dimensional space. The vector from "man" to "woman" is the same as the vector from "king" to "queen."

**Practical Introduction: The Vertex AI API**

After establishing the theory, this module gets practical. It introduces the **Google Cloud Vertex AI** platform as a managed machine learning environment. You learn:

1.  **Setting up the Environment:** How to enable the Vertex AI API in a Google Cloud project and launch a managed Vertex AI Notebook (a JupyterLab instance).
2.  **Authentication:** How to configure your environment with the necessary permissions (e.g., service accounts) so your code can make calls to Google's powerful, pre-trained models.
3.  **Your First API Call:** You will use the `google-cloud-aiplatform` SDK to initialize a client and get your first embedding. The code is shown to be remarkably simple:
    ```python
    from vertexai.preview.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(["Hello world!", "This is a test."])
    # The 'embeddings' variable now holds a list of vectors
    # e.g., [[0.01, -0.42, ..., 0.91], [0.33, 0.09, ..., -0.12]]
    ```
4.  **Understanding the Output:** The module concludes by having you inspect this output. You'll see that each piece of text, whether a single word or an entire paragraph, is converted into a vector of a fixed length (e.g., 768 dimensions for the `textembedding-gecko` model). This vector is now a mathematical "summary" of the text's semantic meaning, and it is the raw material for all the applications in the subsequent modules.

</details>

<details>
<summary><strong>2. Understanding Text Embeddings</strong></summary>

While the first module introduced *what* an embedding is and how to generate one via an API, this module dives deep into the *theory* of *how* these embeddings are created and the *properties* that make them so powerful. It moves from "black-box user" to "informed practitioner" by exploring the models and concepts that operate under the hood.

**Part 1: How are Embeddings Learned? (The Models)**

This section explores the classic algorithms that defined the field of embedding generation.

  * **Word2Vec (Prediction-Based):** Developed at Google by Tomas Mikolov, Word2Vec was a breakthrough. It proposed that you could learn dense vectors by training a simple neural network on a "fake" task.

      * **CBOW (Continuous Bag-of-Words):** The "fake task" is: **"Predict the middle word from its surrounding context words."** The model takes the (averaged) vectors for `["The", "quick", "___", "fox", "jumps"]` as input and is trained to output the vector for `"brown"`.
      * **Skip-gram:** The "fake task" is the inverse: **"Predict the surrounding context words from the middle word."** The model takes the vector for `"brown"` as input and is trained to output the vectors for `["The", "quick", "fox", "jumps"]`.
      * **The Key Insight:** The neural network itself is not the goal. The *real* product is the *hidden layer* (the "embedding lookup table") of that network. After training on billions of sentences, this hidden layer becomes a powerful map where each row (corresponding to a word) is the 300-dimensional vector for that word.
      * **Negative Sampling:** The module also explains this crucial optimization. Instead of forcing the model to predict the correct word from all 50,000 possible vocabulary words (a massive softmax calculation), negative sampling simplifies the problem. It trains a binary classifier: "For the input `"brown"`, is `"fox"` a real context word? (Yes/1)"... "Is `"banana"` a real context word? (No/0)". This is *vastly* more efficient to compute.

  * **GloVe (Global Vectors for Word Representation):** Developed at Stanford, this model takes a different approach. Instead of a "fake" predictive task, it's based on *co-occurrence statistics*.

      * **The Process:**
        1.  First, it scans the entire corpus and builds a giant **co-occurrence matrix (X)**. `X_ij` holds the number of times word `i` appeared in the context of word `j`.
        2.  It then trains a model whose objective is to learn vectors `w` such that the dot product of two word vectors (`w_i` • `w_j`) is proportional to the *logarithm* of their co-occurrence count (`log(X_ij)`).
      * **The Difference:** Word2Vec is a *predictive* model (it only sees local context windows). GloVe is a *count-based* model (it has a "global" view of the entire corpus's statistics). Both produce high-quality embeddings.

**Part 2: From Word Embeddings to Sentence Embeddings**

This section addresses a critical next step. Word2Vec and GloVe give you a vector for *one word*. How do you get a single vector for an *entire sentence* or *paragraph*?

  * **Simple (but flawed) Methods:**

      * **Vector Averaging:** The simplest way is to get the vector for every word in the sentence and just *average them* (`(vec("the") + vec("quick") + vec("brown") + vec("fox")) / 4`).
      * **Problem:** This is surprisingly effective as a baseline, but it's a "bag-of-vectors." It loses all word order and syntax. `"Man bites dog"` and `"Dog bites man"` would have identical averaged embeddings.

  * **Advanced (Contextual) Methods:** This is the leap to modern models.

      * **RNNs/LSTMs:** Recurrent Neural Networks (RNNs) and their more advanced variant, Long Short-Term Memory (LSTMs), were designed to process sequences. They read the sentence word-by-word, and a "hidden state" vector accumulates meaning as it goes. The *final* hidden state can be used as the sentence embedding.
      * **Transformers (BERT, T5, etc.):** This is the current state-of-the-art and what powers models like Vertex AI's `textembedding-gecko`.
          * **The "Attention" Mechanism:** Unlike an RNN which reads left-to-right, a Transformer model (specifically the *encoder* part) reads the *entire sentence all at once*. Using a mechanism called "self-attention," it can weigh the importance of every other word *for* a given word.
          * **Contextual vs. Static Embeddings:** This is the *most important concept* of the module.
              * In Word2Vec, the vector for `"bank"` is *static*—it's the *same* in `"river bank"` and `"financial bank"`.
              * In a Transformer model like BERT, the output embedding for `"bank"` is *contextual*. The self-attention mechanism will see `"river"` and produce a final embedding for `"bank"` that is very different from the one it would produce if it saw `"financial"`.
          * **How BERT Creates a Sentence Embedding:** BERT is pre-trained to use a special `[CLS]` (classification) token at the beginning of every sentence. The final, output embedding corresponding to this *one* token is designed to be a holistic summary of the entire sentence, suitable for classification or other tasks. Other models, like Sentence-BERT (S-BERT), are specifically fine-tuned so that their *averaged* output vectors (a process called "mean pooling") are semantically meaningful and directly comparable with cosine similarity.

**Part 3: Properties of the Embedding Space**

The module concludes by discussing what makes a "good" embedding space.

  * **Semantic Similarity:** This is the primary property. It's measured by **Cosine Similarity**.
      * **Formula:** $similarity = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$
      * **Explanation:** This formula measures the *angle* between two vectors, not their *magnitude*.
          * A score of **1.0** means the vectors point in the *exact same direction* (perfectly similar).
          * A score of **0.0** means the vectors are *orthogonal* (no relationship).
          * A score of **-1.0** means the vectors are *opposite* (perfectly dissimilar).
      * You learn that sentence embeddings are designed so that `"How is the weather?"` and `"What's the forecast today?"` will have a cosine similarity close to 1.0, while `"How is the weather?"` and `"My cat is blue"` will have a similarity close to 0.0.
  * **Normalization:** You learn that for similarity tasks, we almost always *normalize* the embeddings (scale them to have a length, or "norm," of 1.0). When vectors are normalized, the (computationally expensive) cosine similarity calculation simplifies to just a (very fast) *dot product*. This is a critical optimization for vector search.

</details>

<details>
<summary><strong>3. Visualizing Embeddings</strong></summary>

This module addresses a major practical and conceptual challenge: text embeddings are 768-dimensional (or more). Humans cannot "see" in 768 dimensions. Our brains are wired for 2D and 3D. The entire purpose of this module is to explore **dimensionality reduction** techniques, which are algorithms that "squish" or "project" high-dimensional data into a low-dimensional space (2D or 3D) that we can plot on a scatter chart.

The goal is to create a 2D plot where *points that are close in 768D space* are also *close on the 2D plot*. This allows us to *visually* inspect our embedding space and see if clusters of semantically similar text are *actually* forming.

**The "Curse of Dimensionality"**

The module first explains *why* this is hard. In high-dimensional spaces, our 3D-intuition breaks down.

  * **Sparsity:** Data points become extremely far apart from each other.
  * **Meaningless Distances:** The difference between the "farthest" and "closest" neighbor to a point can become almost negligible, making "nearest neighbor" searches difficult.

**Technique 1: PCA (Principal Component Analysis)**

  * **What it is:** A *linear* dimensionality reduction technique.
  * **How it Works (Conceptual):** PCA's goal is to find the *directions* in the data that capture the *most variance*.
    1.  It finds the "First Principal Component," which is the single 768D axis (a linear combination of the original 768 axes) along which the data is most "spread out."
    2.  It then finds the "Second Principal Component," which is the *next* most-variant axis, with the constraint that it must be *orthogonal* (at a 90-degree angle) to the first one.
    3.  It continues this for all 768 dimensions.
    4.  To get a 2D visualization, you simply *project* your data points onto the *first two* principal components (PC1 and PC2) and plot those new (x, y) coordinates.
  * **Analogy:** Imagine a 3D cloud of points shaped like a pancake. PCA would find that PC1 is the axis going along the "length" of the pancake, PC2 is the axis along the "width," and PC3 is the axis through the "thickness." By plotting just PC1 and PC2, you get a 2D view that preserves 99% of the pancake's structure.
  * **Pros:**
      * Very fast to compute.
      * Deterministic (it gives the same result every time).
      * Easy to understand; the axes are "explainable" (they are linear combinations of the original features).
  * **Cons:**
      * It's **linear**. It cannot capture complex, non-linear "manifold" structures. If your data is shaped like a "Swiss Roll" or a "corkscrew," PCA will just flatten it, destroying the local relationships.
      * It optimizes for *variance*, not for *local neighborhood preservation*. This means two points that were "close" in 768D might end up "far" on the 2D plot if that's what's required to maximize the overall variance.

**Technique 2: t-SNE (t-Distributed Stochastic Neighbor Embedding)**

  * **What it is:** The *star* of embedding visualization. A powerful *non-linear* technique.
  * **How it Works (Conceptual):** This is a much more complex, iterative algorithm.
    1.  **High-D Probabilities:** For every point `i`, it models a probability distribution of picking another point `j` as its "neighbor." This distribution is based on a Gaussian (bell curve) centered at `i`. Points *close* to `i` have a high probability; points *far* from `i` have a near-zero probability.
    2.  **Low-D Probabilities:** It then scatters all the points *randomly* onto a 2D plane. It models a *second* probability distribution in this 2D space, but this time using a *t-distribution* (which has "heavier tails" than a Gaussian, helping to prevent points from "crowding" too much in the center).
    3.  **Optimization (Gradient Descent):** The algorithm's goal is to make the 2D probability distribution map look *as similar as possible* to the original 768D probability distribution map. It iteratively "nudges" the 2D points around, "pulling" points that are neighbors in 768D *together* and "pushing" points that are *not* neighbors *apart*.
  * **The "Perplexity" Parameter:** This is the main knob you tune. It's a "soft" way of telling the algorithm how many "effective neighbors" to consider for each point. A low perplexity (e.g., 5) focuses on preserving very, very local structure. A high perplexity (e.g., 50) tries to preserve a more global structure.
  * **Pros:**
      * **Excellent at revealing local structure.** It creates beautiful, clear, and intuitive "islands" or "clusters" of data. It is fantastic for *cluster discovery*.
  * **Cons:**
      * **Extremely Slow:** It is computationally expensive ($O(N \log N)$ or $O(N^2)$) and can take hours for large datasets.
      * **Non-Deterministic:** Running it twice (with different random starting positions) will produce two *different-looking* plots (though the cluster groupings are usually preserved).
      * **Global Distances are MEANINGLESS:** This is the *most critical* caveat. The *size* of a cluster on a t-SNE plot means nothing. The *distance between two separate clusters* on the plot also means *nothing*. It is a common and very dangerous mistake to say "Cluster A is twice as big as Cluster B" or "Cluster A is farther from B than C." t-SNE *only* preserves *local* neighborhoods.

**Technique 3: UMAP (Uniform Manifold Approximation and Projection)**

  * **What it is:** A "modern" non-linear technique that is often seen as a successor to t-SNE.
  * **How it Works (Conceptual):** It's based on advanced topological data analysis. It builds a "fuzzy graph" representing the connectivity of the high-dimensional data, then tries to find the most "equivalent" graph in the 2D space.
  * **Pros:**
      * **Much Faster** than t-SNE.
      * Often produces *better* cluster separation.
      * Arguably *better at preserving global structure*. The distances *between* clusters in a UMAP plot can be *more* (though not perfectly) meaningful than in t-SNE.
  * **Cons:**
      * Newer and more complex mathematically.

**Practical Visualization Tools**

The module concludes by showing how to use these algorithms practically.

  * **`sklearn`:** Using `sklearn.decomposition.PCA` and `sklearn.manifold.TSNE` to get the 2D coordinates.
  * **`matplotlib` / `seaborn`:** Using standard plotting libraries to create scatter plots.
  * **Interactive Plotting:** The *real* power comes from interactive tools like **Plotly** or **Altair**, which allow you to *hover your mouse* over a point on the 2D plot and see a tooltip displaying the *original text* it represents. This is how you *interpret* the clusters.
  * **Embedding Projector:** This module will almost certainly introduce the **TensorFlow Embedding Projector**, a powerful, open-source web tool from Google. You can upload your embedding vectors and labels, and it will run PCA, t-SNE, and UMAP for you in an interactive 3D environment, allowing you to fly through, search, and label your embedding space.

</details>

<details>
<summary><strong>4. Applications of Embeddings</strong></summary>

Now that we have a solid theoretical understanding of embeddings and how to visualize them, this module gets to the heart of the course: *using* them. The key insight is that embeddings transform unstructured, messy text into a high-dimensional, but perfectly structured, numerical matrix `(N_samples, D_features)`. This matrix is the "lingua franca" of machine learning. We can now feed this `X` matrix into *any* classic ML model to perform powerful tasks.

This module focuses on three key applications, as listed in the course description: classification, clustering, and outlier detection.

**Application 1: Text Classification**

  * **What is it?** A *supervised* learning task. The goal is to assign a pre-defined label or category to a piece of text.
  * **Examples:**
      * **Spam Detection:** Labeling an email as `"spam"` or `"not_spam"`.
      * **Sentiment Analysis:** Labeling a product review as `"positive"`, `"negative"`, or `"neutral"`.
      * **Topic Categorization:** Labeling a news article as `"sports"`, `"technology"`, or `"politics"`.
      * **Intent Recognition:** Labeling a user's chat message as `"check_balance"`, `"transfer_funds"`, or `"request_help"`.
  * **The Pipeline:**
    1.  **Gather Labeled Data:** You need a training set of `(text, label)` pairs.
          * `("This movie was awesome!", "positive")`
          * `("I hated every minute of it.", "negative")`
    2.  **Generate Embeddings:** You pass *all* the training texts through the Vertex AI Text-Embeddings API.
    3.  **Create `X` and `y`:**
          * `X_train` is now a matrix of `(N_samples, 768_dimensions)`—our embeddings.
          * `y_train` is a vector of `(N_samples,)`—our labels (`["positive", "negative", ...]`).
    4.  **Train a Classifier:** This is now a "classic" ML problem. You can feed `X_train` and `y_train` into any `sklearn` model.
          * **Logistic Regression:** A fantastic baseline. It's fast, interpretable, and works very well on high-dimensional data. It will find a *linear* "hyperplane" in the 768D space that best separates the "positive" and "negative" vectors.
          * **Support Vector Machine (SVM):** Another powerful choice, often even better. SVMs (with a linear kernel) are excellent at finding the "maximum margin" hyperplane, which is very robust.
          * **Random Forest / XGBoost:** Tree-based models can also work, but they sometimes struggle with the "smooth" continuous nature of embedding spaces. They are better at handling tabular data with discrete features.
          * **Simple Neural Network:** You can build a small `Keras` / `TensorFlow` model with a few `Dense` layers. This is essentially creating a custom "head" on top of the pre-trained embedding model.
    5.  **Inference:** To classify a *new, unseen* piece of text, the process is simple:
        1.  Get its embedding from the Vertex AI API.
        2.  Call `model.predict(new_embedding)`.
  * **Why it's Better:** A classifier built on TF-IDF would be confused by `"This film was incredible"` and `"This movie was outstanding"` if it hadn't seen those exact words. A classifier built on *embeddings* will work perfectly, because the *vectors* for those two sentences are already extremely close in the embedding space.

**Application 2: Text Clustering**

  * **What is it?** An *unsupervised* learning task. You have a large collection of *unlabeled* text, and you want to discover *natural groupings* or *themes* within it.
  * **Examples:**
      * Grouping 10,000 customer support tickets to find the "Top 5 most common issues."
      * Clustering news articles to automatically create a "Topics" section.
      * Analyzing open-ended survey responses (`"What did you like least?"`) to find common complaints.
  * **The Pipeline:**
    1.  **Gather Unlabeled Data:** You just have a big list of texts.
    2.  **Generate Embeddings:** Convert all texts into a vector matrix `X`.
    3.  **Run a Clustering Algorithm:**
          * **K-Means:**
              * **How it works:** You must first *specify* the number of clusters, `k` (e.g., `k=5`). The algorithm then finds `k` "centroids" (center points) in the 768D space and assigns each document to its *nearest* centroid.
              * **Finding `k`:** You can use the "Elbow Method" (plotting inertia vs. `k`) to find an optimal `k`.
              * **Distance Metric:** K-Means uses Euclidean distance. Because embeddings are best compared with *cosine similarity*, a crucial pre-processing step is to **normalize** all your embedding vectors (make them unit-length). On normalized vectors, minimizing Euclidean distance is *mathematically equivalent* to maximizing cosine similarity.
          * **DBSCAN (Density-Based Spatial Clustering):**
              * **How it works:** A more advanced algorithm that doesn't require you to set `k`. Instead, it finds "dense" regions in the data. It's great because it can find clusters of *any* shape (not just spherical ones like K-Means) and, most importantly, it automatically *identifies noise points (outliers)* that don't belong to *any* cluster.
    4.  **Interpret Clusters:** The model outputs cluster labels (e.g., `"cluster_0"`, `"cluster_1"`, etc.). Your job is to *interpret* them. You do this by sampling 10-20 texts from each cluster and reading them.
          * `cluster_0` texts: `"My password won't reset", "I can't log in", "Forgot my username"` -> You label this cluster **"Account Access Issues"**.
          * `cluster_1` texts: `"The app is crashing", "The button is broken", "Error message 503"` -> You label this cluster **"App Bugs / Errors"**.

**Application 3: Outlier Detection (Anomaly Detection)**

  * **What is it?** The task of finding the "odd one out." Identifying documents that are semantically *different* from the rest of the dataset.
  * **Examples:**
      * Finding a single fake product review in a sea of 1,000 real ones.
      * Detecting an off-topic post on a technical support forum.
      * Identifying "topic drift" in a live feed of news articles.
  * **Methods:**
    1.  **DBSCAN:** As mentioned above, DBSCAN automatically labels "noise" points. These are your outliers. This is one of the most effective methods.
    2.  **Local Outlier Factor (LOF):** An algorithm that computes an "outlier score" for each point. It measures how "isolated" a point is from its local neighborhood. High scores = outliers.
    3.  **Centroid-Based (Simplest Method):**
          * Assume your dataset is *mostly* "normal" (e.g., product reviews for a *specific* product).
          * Generate embeddings for all 10,000 reviews.
          * Compute the *centroid* (the average vector) of all 10,000 embeddings. This single vector represents the "average review" or the "center of the main topic."
          * For *every* review, calculate its cosine similarity to this main centroid.
          * Sort the reviews by this similarity score, from *lowest to highest*.
          * The reviews at the *bottom* of the list (e.g., similarity < 0.3) are your most likely outliers. You'd inspect them and find they might be spam, for the *wrong product*, or just bizarre.

</details>

<details>
<summary><strong>5. Text Generation with Vertex AI</strong></summary>

This module represents a significant pivot. After four modules focused on *understanding* and *analyzing* text with embeddings (`text -> vector`), this module switches to *generating* new text (`prompt -> new text`). It introduces the other side of the Vertex AI suite: Large Language Models (LLMs) like PaLM and Gemini.

The key to this module is understanding *how* an LLM generates text and *how* we can control its output to make it more creative or more factual.

**How LLMs Generate Text: Autoregressive Sampling**

The module first explains that LLMs are "autoregressive" models. This means they generate text *one token (word or sub-word) at a time*.

1.  **Input:** The user provides a prompt: `["The", "quick", "brown", "fox"]`
2.  **Prediction:** The LLM processes this prompt and produces a *probability distribution* over its entire vocabulary (e.g., 50,000+ tokens) for *what the very next token should be*.
      * `P("jumps") = 0.85`
      * `P("ran") = 0.09`
      * `P("ate") = 0.03`
      * ...
      * `P("banana") = 0.000001`
3.  **Sampling:** The model *chooses* a token from this distribution (e.g., it samples and gets `"jumps"`).
4.  **Append:** This new token is *appended* to the input. The model's new input is now: `["The", "quick", "brown", "fox", "jumps"]`
5.  **Repeat:** The process repeats. The model predicts the *next* token:
      * `P("over") = 0.9`
      * `P("into") = 0.04`
      * ...
6.  This loop continues (`...fox jumps over the lazy dog`) until the model predicts a special `[END_OF_SEQUENCE]` token or hits a maximum output length.

**The "Sampling" Problem: Greedy vs. Stochastic**

How does the model *choose* the next token from the probability list?

  * **Greedy Decoding:** Always pick the token with the *highest probability*.
      * **Result:** This is deterministic, safe, and coherent.
      * **Problem:** It's also incredibly *boring* and *repetitive*. It has no "creativity" or "flair." It can easily get stuck in loops (`"I am a robot. I am a robot. I am a robot..."`).
  * **Stochastic (Random) Sampling:** We need to *sample* from the distribution. But if we sample from the *entire* distribution, we have a small (but real) chance of picking `"banana"` in our "fox" sentence, leading to nonsensical (or "unhinged") output.

This is where the "knobs" for controlling generation come in. This module, as per the course description, focuses on `temperature`, `top-k`, and `top-p`.

**Parameter 1: `temperature`**

  * **What it is:** A *scaling factor* that is applied to the raw output scores (logits) *before* the softmax function calculates the probabilities.
  * **How it Works:** `probabilities = softmax(logits / temperature)`
  * **`temperature` > 1.0 (e.g., 1.5):** *Increases Randomness.*
      * It divides the logits, making them *closer together*. The probability distribution becomes *flatter*.
      * `P("jumps")=0.85` might become `P("jumps")=0.6`, `P("ran")=0.2`, `P("ate")=0.1`.
      * **Result:** The model becomes more "creative," "daring," and "surprising." It's more likely to pick less-common words.
      * **Use Case:** Brainstorming, marketing copy, writing poetry.
  * **`temperature` < 1.0 (e.g., 0.2):** *Decreases Randomness.*
      * It divides the logits by a small number, which *exaggerates* the differences. The distribution becomes *sharper* or "peakier."
      * `P("jumps")=0.85` might become `P("jumps")=0.999`.
      * **Result:** The model becomes more "deterministic," "conservative," and "factual." It will stick very closely to the most obvious, high-probability words.
      * **`temperature` -> 0:** Becomes identical to greedy decoding.
      * **Use Case:** Factual Q&A, code generation, summarization of a legal document.

**Parameter 2: `top-k` Sampling**

  * **What it is:** A simple, *hard cutoff* based on *rank*.
  * **How it Works:**
    1.  You set `k` (e.g., `k=40`).
    2.  The model looks at its full probability distribution.
    3.  It *discards* all tokens *except* for the `k` most probable ones.
    4.  It then *renormalizes* the probabilities of just those `k` tokens (so they sum to 1.0).
    5.  Finally, it samples (using temperature) from this *new, smaller* distribution.
  * **Result:** This effectively creates a "dynamic vocabulary" for each generation step. It *guarantees* that the model can *never* choose a truly bizarre, low-probability word (like `"banana"`), because that word will almost certainly not be in the `top-k` list.
  * **Problem:** It's not *adaptive*. If the model is *very certain* (`P("jumps")=0.99`), `k=40` still forces it to consider 39 other (mostly bad) options. If the model is *very uncertain* and has 100 *reasonable* next words (each with \~1% probability), `k=40` would *arbitrarily cut off* 60 of those reasonable options.

**Parameter 3: `top-p` (Nucleus) Sampling**

  * **What it is:** A more intelligent, *adaptive* cutoff based on *cumulative probability*. This is often the preferred method.
  * **How it Works:**
    1.  You set `p` (e.g., `p=0.95`).
    2.  The model sorts all tokens by probability, from highest to lowest.
    3.  It goes down the list, *summing up their probabilities* until the sum reaches `p`.
    4.  This *set* of tokens (the "nucleus") is kept. All other tokens are discarded.
    5.  It renormalizes and samples from this nucleus.
  * **The Adaptive "Magic":** This solves the `top-k` problem.
      * **Case 1 (High Certainty):**
          * `P("jumps") = 0.98`, `P("ran") = 0.01`, ...
          * With `p=0.95`, the nucleus contains *only* the token `"jumps"`. The model will choose it.
      * **Case 2 (Low Certainty):**
          * `P("ate")=0.15, P("ran")=0.12, P("saw")=0.10, ... [and so on]`
          * With `p=0.95`, the nucleus will be *much larger*, containing maybe 20 or 30 tokens until their probabilities add up to 0.95.
  * **Result:** The model is *adaptive*. When it's "sure," it acts like greedy. When it's "unsure," it considers a wide range of *reasonable* options.

**Practical Use with Vertex AI**

This module concludes by showing how to call the Vertex AI LLM API (e.g., PaLM 2 or Gemini) and pass in these parameters:

```python
from vertexai.preview.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@001")
response = model.predict(
    "The quick brown fox...",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=128
)
print(response.text)
# Output: "...jumps over the lazy dog."
```

This gives you practical, hands-on experience in "prompt engineering" and "output tuning," shifting the LLM from a generic chatbot to a tool fine-tuned for a specific task.

</details>

<details>
<summary><strong>6. Building a Q&A System Using Semantic Search</strong></summary>

This is the capstone module of the course. It synthesizes *everything* you have learned:

1.  **Text Embeddings** (from Modules 1 & 2)
2.  **Vector Comparison** (Cosine Similarity, from Module 2)
3.  **LLM Generation** (from Module 5)

The goal is to build a **Retrieval-Augmented Generation (RAG)** system. This is one of the most powerful and common applications of LLMs in the enterprise today.

**The Problem: Why RAG?**

This module starts by defining the core weaknesses of "vanilla" LLMs:

1.  **Hallucinations:** They confidently "make up" facts and sources that do not exist.
2.  **Knowledge Cutoff:** An LLM trained on data up to December 2023 has no knowledge of events, products, or information from 2024.
3.  **Private Data:** The LLM was not trained on *your* company's internal wiki, *your* project's technical documentation, or *your* HR policy PDFs.

You cannot ask a "vanilla" LLM, `"What is my company's policy on remote work?"` It will either say "I don't know" or, worse, *hallucinate* a policy.

**The Solution: Retrieval-Augmented Generation (RAG)**

The RAG pattern solves this by giving the LLM an "open-book exam." Instead of asking the LLM to *recall* the answer from its (outdated/incomplete) memory, you **(1) Retrieve** the relevant information first and then **(2) Augment** the LLM's prompt with that information, asking it to **(3) Generate** an answer *based only on the provided text*.

This module breaks the RAG process into two phases:

**Phase 1: Indexing (The "Offline" Step)**

This is the one-time (or periodic) setup where you build your "knowledge base."

1.  **Load Documents:** Ingest your private data (e.g., all 50 HR policy PDFs, 200 technical docs, 1000 wiki pages).
2.  **Chunk Documents:** This is a *critical* step. You cannot create a single embedding for a 100-page PDF. The embedding would be a meaningless "average" of 100 pages. You must *split* the documents into small, semantically-meaningful "chunks." For example:
      * Split by paragraph.
      * Split into 500-token chunks with a 50-token overlap (to avoid cutting a sentence in half).
3.  **Embed Chunks:** You loop through your *entire collection* of (e.g.,) 10,000 chunks and get an embedding for *each one* using the Vertex AI Text-Embeddings API.
4.  **Store in a Vector Database:** You now have 10,000 vectors. You need to store them in a way that allows for *fast search*. You can't use a `for` loop to check 10,000 cosine similarities; it's too slow. This is where a **Vector Database** comes in.

**Vector Search & ScaNN (Scalable Nearest Neighbors)**

The course introduces the challenge of **Nearest Neighbor (NN) Search**: "Given a query vector, find the `k` (e.g., `k=5`) vectors in my database that have the highest cosine similarity."

  * **Brute-Force Search:** `O(N*D)` complexity. Comparing 1 query vector against 1 million 768-D vectors is slow (1 million calculations).
  * **Approximate Nearest Neighbor (ANN) Search:** This is the solution. ANN algorithms trade *perfect 100% accuracy* for *massive 10,000x speed*. They might return the 1st, 2nd, 3rd, 5th, and 7th closest neighbors (missing the 4th and 6th), but this is "good enough" for RAG and is virtually instantaneous.

This is where the course introduces **ScaNN (Scalable Nearest Neighbors)**:

  * **What it is:** Google's open-source, state-of-the-art ANN library.
  * **How it Works (Conceptually):** It's incredibly clever.
    1.  **Vector Quantization:** It *compresses* the large, 768-dimension `float32` vectors into much smaller, `int8` representations. This is "lossy" but saves huge amounts of memory.
    2.  **Clustering/Partitioning:** It pre-partitions all 10,000 vectors into (e.g.,) 100 clusters. When a new query vector comes in, it first figures out which 3 clusters are *closest* to the query, and *only* searches the vectors within those 3 clusters (e.g., 300 vectors) instead of all 10,000.
  * **Vertex AI Matching Engine:** The course explains that you don't need to build and host ScaNN yourself. **Vertex AI Matching Engine** (formerly Vector Search) is the *fully managed, auto-scaling, serverless* version. You just create an "index," upload your vectors via an API, and Google handles the rest.

**Phase 2: Querying (The "Online" Step)**

This is what happens in real-time when a user asks a question.

1.  **User Question:** `(e.g., "What is the policy on international remote work?")`

2.  **Embed Query:** Get the embedding for *this question* using the Vertex AI API: `query_vector = embed("What is the policy...")`.

3.  **Semantic Search:** Send this `query_vector` to your Vertex AI Matching Engine index and ask for the `k=5` nearest neighbors.

4.  **Retrieve Chunks:** The index returns the IDs of the top 5 chunks (e.g., `[chunk_901, chunk_42, chunk_1234]`). You look up these chunks from your original text store.

      * `chunk_901`: "International remote work is handled on a case-by-case basis and requires VP approval..."
      * `chunk_42`: "Our remote work policy, established in 2023, allows..."
      * `chunk_1234`: "Employees working remotely must maintain a secure..."

5.  **Augment (Prompt Engineering):** This is the RAG "magic." You *dynamically build a new prompt* for the LLM:

    ```
    You are a helpful HR assistant. Answer the user's question based *only* on the context provided below. If the answer is not in the context, say "I am sorry, I do not have that information."

    [START CONTEXT]
    - chunk_901: "International remote work is handled on a case-by-case basis and requires VP approval..."
    - chunk_42: "Our remote work policy, established in 2023, allows..."
    - chunk_1234: "Employees working remotely must maintain a secure..."
    [END CONTEXT]

    USER QUESTION:
    What is the policy on international remote work?

    ANSWER:
    ```

6.  **Generate:** You send this *entire prompt* (context + question) to the Vertex AI LLM (Gemini or PaLM).

7.  **Final Answer:** The LLM, now "grounded" in facts, will not hallucinate. It will synthesize an answer directly from the context: `(e.g., "According to the policy, international remote work is handled on a case-by-case basis and requires VP approval.")`

This capstone project provides a complete, end-to-end framework for building powerful, accurate, and safe Q&A systems using your own private data.

</details>

-----

## Acknowledgement

This repository is for personal, educational, and learning purposes only. It contains personal notes and code implementation based on the official course.

All course content, materials, and intellectual property rights are held by **DeepLearning.AI** and **Google Cloud**. I do not claim any ownership of the original course material. Please enroll in the official course on Coursera to access the full and certified content.