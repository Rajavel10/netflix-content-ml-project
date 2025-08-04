# Netflix Content Analysis & Clustering Project

> **A comprehensive unsupervised machine learning project to explore, analyze, and cluster Netflix’s global catalog for insightful content segmentation and data-driven discovery.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features & Columns](#features--columns)
- [Project Pipeline](#project-pipeline)
- [Key Results & Insights](#key-results--insights)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Netflix offers a vast collection of movies, TV shows, and documentaries from around the globe. Navigating and recommending content in such a large, diverse catalog poses significant analytical and business challenges. 

This project uses Python (with pandas, scikit-learn, etc.) to:
- Cleanse and preprocess Netflix title data.
- Engineer advanced features from both structured and textual data.
- Apply unsupervised learning (clustering) models to discover natural content groupings.
- Evaluate clusters, visualize distributions, and interpret their business significance.

The project demonstrates a real-world, end-to-end data science approach from raw data to actionable insights—ideal for portfolios and practical demonstration.

---

## Dataset

- **Source:** Kaggle Netflix Dataset 
- **Shape:** 8,800+ entries, multiple columns
- **Included:** Movies and TV Shows with details (title, type, director, cast, country, date added, release year, rating, duration, genres, description)

*Dataset file(s) are included for reference and reproducibility.*

---

## Features & Columns

**Key Variables:**
- `title`: NETFLIX MOVIES AND TV SHOWS CLUSTERING DATASET
- `type`: Movie or TV Show
- `director`, `cast`, `country`
- `date_added`, `release_year`
- `rating`: Age rating (e.g., TV-MA, PG-13)
- `duration`: Run time in minutes or number of seasons
- `listed_in`: Genres
- `description`: Brief summary

**Engineered Features:**
- `duration_num`: Numeric duration
- `cast_count`: Number of main actors
- `release_decade`: Decade when content was released
- TF-IDF vectors: Top genres/topics extracted from content descriptions

---

## Project Pipeline

1. **Data Cleaning**
   - Imputation of missing categorical values (e.g., country, cast)
   - Capping outliers for features like duration
2. **Text Processing**
   - Lowercasing, removing punctuation, stopword elimination
   - Lemmatization and TF-IDF feature extraction from `listed_in` and `description`
3. **Feature Engineering**
   - New columns for content decade, cast size, and more
   - Label encoding and one-hot encoding for categorical columns
4. **Scaling & Dimensionality Reduction**
   - Standardization of numeric features
   - (Optional) PCA for visualization and efficiency
5. **Clustering**
   - KMeans, Agglomerative, and DBSCAN explored
   - Hyperparameter tuning via silhouette score and elbow method
6. **Visualization**
   - Silhouette score histograms, cluster distributions, and content cluster profiles
7. **Interpretation**
   - Linking clusters back to meaningful content types and business goals

---

## Key Results & Insights

- Identified distinct clusters such as “Family Content,” “International Movies,” “Short-format Series,” etc.
- Demonstrated how clusters can power targeted recommendations, catalog curation, and marketing campaigns.
- Showed the utility of text features (genre/topic scores) in distinguishing subtle content differences.

---

## How to Run

git clone https://github.com/Rajavel10/netflix-content-ml-project.git
cd netflix-content-ml-project


2. **Install dependencies:**
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, nltk
- (Install via `pip install -r requirements.txt` if provided)

3. **Open the notebook:**
- Launch `Unsupervised ML - Netflix Movies and TV Shows Clustering.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

4. **Run all cells**
- Follow along with markdown explanations.
- Modify parameters or models as desired.

---

## Repository Structure

├── NETFLIX MOVIES AND TV SHOWS CLUSTERING DATASET.csv # Netflix dataset (format may vary)

├── Unsupervised_ML_Netflix_Movies_and_TV_Shows_Clustering.ipynb # Main analysis and clustering notebook

├── best_model.pkl  (Saved clustering model)

├── requirements.txt  (Dependence list)

└── README.md 



---

## Future Work

- Deploy clustering as a microservice for live recommendation use.
- Extend to supervised models for rating/classification tasks.
- Deeper content analysis with NLP models (BERT, word embeddings).
- Automated dashboards for ongoing catalog monitoring.

---

## Acknowledgments

- Original Netflix dataset from Kaggle
- Open-source Python packages and community documentation.
- Project inspiration: Modern recommender systems and streaming analytics.

---

**Questions? Contributions?**  
Open an issue or pull request—collaboration is welcome!

---
