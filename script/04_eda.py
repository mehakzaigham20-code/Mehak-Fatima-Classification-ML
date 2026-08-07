import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/preprocessed_news.csv")

# Category distribution

counts = df["Category"].value_counts()

plt.figure(figsize=(10, 6))
counts.plot(kind="bar")
plt.title("News Categories Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/category_distribution.png", dpi=300)
plt.show()

# Title length

plt.figure(figsize=(10, 6))
plt.hist(df["Title_Length"], bins=30)
plt.title("Title Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/title_length_distribution.png", dpi=300)
plt.show()

# Word count

plt.figure(figsize=(10, 6))
plt.hist(df["Word_Count"], bins=30)
plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/word_count_distribution.png", dpi=300)
plt.show()

print("✅ EDA completed.")