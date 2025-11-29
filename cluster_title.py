from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_titles(title_string):
    titles = [t.strip() for t in title_string.split("\n") if t.strip()]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(titles)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1
    )

    labels = clusterer.fit_predict(embeddings)

    clusters = {}
    for title, label in zip(titles, labels):
        clusters.setdefault(label, []).append(title)

    return clusters

def summarize_cluster(cluster_titles):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(cluster_titles)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    top_keywords = [keywords[i] for i in scores.argsort()[-3:][::-1]]
    return ", ".join(top_keywords)

if __name__ == "__main__":
    giant_string = ""

    clusters = cluster_titles(giant_string)

    for label, cluster in clusters.items():
        if label == -1:
            print("\nOUTLIERS:")
        else:
            print(f"\nCLUSTER {label} ({summarize_cluster(cluster)}):")

        for title in cluster:
            print(" -", title)