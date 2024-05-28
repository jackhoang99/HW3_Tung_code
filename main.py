import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ssl
import pandas as pd

# Bypass SSL verification for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")


def read_documents(documents_path):
    document_files = [
        os.path.join(documents_path, file)
        for file in os.listdir(documents_path)
        if file.endswith(".txt")
    ]
    documents = []
    for file in document_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                documents.append(f.read())
        except UnicodeDecodeError:
            with open(file, "r", encoding="latin-1") as f:
                documents.append(f.read())
    return document_files, documents


def tokenize_documents(documents):
    return [nltk.word_tokenize(doc) for doc in documents]


def compute_token_counts(tokenized_documents):
    token_counts = [len(doc) for doc in tokenized_documents]
    total_tokens = sum(token_counts)
    unique_tokens = len(set(token for doc in tokenized_documents for token in doc))
    return token_counts, total_tokens, unique_tokens


def remove_stop_words(tokenized_documents):
    stop_words = set(stopwords.words("english"))
    stop_words = set(nltk.word_tokenize(" ".join(stop_words)))
    filtered_documents = [
        [word for word in doc if word.lower() not in stop_words]
        for doc in tokenized_documents
    ]
    return filtered_documents, list(stop_words)


def compute_filtered_token_counts(filtered_documents):
    filtered_token_counts = [len(doc) for doc in filtered_documents]
    total_filtered_tokens = sum(filtered_token_counts)
    unique_filtered_tokens = len(
        set(token for doc in filtered_documents for token in doc)
    )
    return filtered_token_counts, total_filtered_tokens, unique_filtered_tokens


def compute_tfidf(documents, stop_words_list):
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=nltk.word_tokenize, stop_words=stop_words_list
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()


def compute_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)


def save_results(
    document_files,
    token_counts,
    filtered_token_counts,
    cosine_sim_matrix,
    tfidf_matrix,
    tfidf_feature_names,
):
    # Token counts
    token_counts_df = pd.DataFrame(
        {
            "Document": document_files,
            "Token Count": token_counts,
            "Filtered Token Count": filtered_token_counts,
        }
    )
    token_counts_df.to_csv("token_counts.csv", index=False)

    # Cosine similarity matrix
    cosine_sim_df = pd.DataFrame(
        cosine_sim_matrix, index=document_files, columns=document_files
    )
    cosine_sim_df.to_csv("cosine_similarity.csv")

    # TF-IDF matrix
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), index=document_files, columns=tfidf_feature_names
    )
    tfidf_df.to_csv("tfidf_matrix.csv")

    print(
        "Results have been saved in 'token_counts.csv', 'cosine_similarity.csv', and 'tfidf_matrix.csv'"
    )
    print("\nTF-IDF Matrix:")
    print(tfidf_df)
    print("\nCosine Similarity Matrix:")
    print(cosine_sim_df)


def main():
    documents_path = "Documents"
    document_files, documents = read_documents(documents_path)
    tokenized_documents = tokenize_documents(documents)
    token_counts, total_tokens, unique_tokens = compute_token_counts(
        tokenized_documents
    )

    print("Token counts for each document:", token_counts)
    print("Total tokens in the entire collection:", total_tokens)
    print("Unique tokens in the entire collection:", unique_tokens)

    filtered_documents, stop_words_list = remove_stop_words(tokenized_documents)
    filtered_token_counts, total_filtered_tokens, unique_filtered_tokens = (
        compute_filtered_token_counts(filtered_documents)
    )

    print(
        "Filtered token counts for each document after stop word removal:",
        filtered_token_counts,
    )
    print(
        "Total filtered tokens in the entire collection after stop word removal:",
        total_filtered_tokens,
    )
    print(
        "Unique filtered tokens in the entire collection after stop word removal:",
        unique_filtered_tokens,
    )

    tfidf_matrix, tfidf_feature_names = compute_tfidf(documents, stop_words_list)
    cosine_sim_matrix = compute_cosine_similarity(tfidf_matrix)

    save_results(
        document_files,
        token_counts,
        filtered_token_counts,
        cosine_sim_matrix,
        tfidf_matrix,
        tfidf_feature_names,
    )


if __name__ == "__main__":
    main()
