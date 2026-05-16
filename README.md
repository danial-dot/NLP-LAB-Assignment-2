# NLP-LAB-Assignment-2
# Text Analysis Project: PDF Text Extraction and Feature Engineering

This project demonstrates a comprehensive workflow for extracting text from a PDF document, performing various text preprocessing steps, and then applying feature extraction techniques like One-Hot Encoding and TF-IDF. Finally, it visualizes the TF-IDF scores using Plotly.

## Table of Contents
1.  [Setup and Installation](#setup-and-installation)
2.  [Project Workflow and Results](#project-workflow-and-results)
    *   [Q1(a): PDF Reading and Text Extraction](#q1a-pdf-reading-and-text-extraction)
    *   [Q1(b): Text Preprocessing](#q1b-text-preprocessing)
    *   [Q1(c): Feature Extraction](#q1c-feature-extraction)
    *   [Q1(d): TF-IDF Scatter Plot Using Plotly](#q1d-tf-idf-scatter-plot-using-plotly)

## Setup and Installation

To run this notebook, you'll need to install the following Python libraries:

```bash
pip install PyPDF2 nltk scikit-learn pandas plotly
```

Additionally, you'll need to download NLTK data. The notebook handles these downloads automatically if not present:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Note**: This notebook assumes a PDF file named `mcs.pdf` is uploaded to the Colab environment's `/content/` directory for text extraction. Please upload your desired PDF accordingly.

## Project Workflow and Results

### Q1(a): PDF Reading and Text Extraction

**Task**: Read a PDF document, extract its text content, display the total page count, and show a sample of the extracted text.

**Steps Performed**:
1.  The `PyPDF2` library was used to open and read the PDF file (`/content/mcs.pdf`).
2.  Text was extracted page by page and concatenated into a single string.
3.  **Result**: The PDF was successfully read, yielding `1048` pages. A sample of the first 500 characters was printed to confirm extraction.

### Q1(b): Text Preprocessing

**Task**: Apply a series of standard text preprocessing techniques to clean and prepare the extracted text for analysis.

**Steps Performed**:
1.  **Lowercase Conversion**: All text was converted to lowercase.
2.  **Number Removal**: All numerical digits were removed using regex (`r'\d+'`).
3.  **Special Symbol Removal**: Non-alphabetic and non-whitespace characters were removed using regex (`r'[^a-zA-Z\s]'`).
4.  **Extra Space Removal**: Multiple spaces were replaced with single spaces, and leading/trailing spaces were trimmed using regex (`r'\s+'`).
5.  **Punctuation Removal**: Standard punctuation marks were removed using `string.punctuation` and regex.
6.  **Tokenization**: The processed text was broken down into individual words (tokens) using `nltk.word_tokenize`.
7.  **Stop Word Removal**: Common English stop words (e.g., 'the', 'is', 'a') were removed using NLTK's `stopwords` corpus. `103636` stop words were identified and removed, leaving `140044` valid words.
8.  **Stemming**: Words were reduced to their root form using NLTK's `PorterStemmer`.
9.  **Lemmatization**: Words were reduced to their base form (lemma) using NLTK's `WordNetLemmatizer`.

### Q1(c): Feature Extraction

**Task**: Transform the preprocessed text into numerical features suitable for machine learning models using One-Hot Encoding and TF-IDF.

**Steps Performed**:
1.  **One-Hot Encoding**: `sklearn.feature_extraction.text.CountVectorizer(binary=True)` was used on the lemmatized text to create a binary presence/absence matrix for each unique word. This resulted in `18683` unique words (features). A sample of the encoded output was displayed.
2.  **TF-IDF (Term Frequency-Inverse Document Frequency)**: `sklearn.feature_extraction.text.TfidfVectorizer` was applied to the lemmatized text. This technique assigns weights to words based on their frequency in the document and across a collection of documents (though in this case, it's a single document). This also yielded `18683` unique words. Sample TF-IDF feature names and their corresponding values were displayed.

### Q1(d): TF-IDF Scatter Plot Using Plotly

**Task**: Visualize the TF-IDF scores using an interactive scatter plot to understand the distribution and importance of words.

**Steps Performed**:
1.  The TF-IDF scores and corresponding words were organized into a pandas DataFrame.
2.  The DataFrame was sorted by TF-IDF score in descending order to identify the most significant words.
3.  `plotly.express.scatter` was used to generate an interactive scatter plot displaying the top `100` words and their TF-IDF scores.
4.  **Result**: An interactive plot was generated, showing words on the x-axis and their TF-IDF scores on the y-axis, providing an intuitive way to explore the most important terms in the document.


