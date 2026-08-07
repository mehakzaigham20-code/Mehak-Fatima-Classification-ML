import pandas as pd
import numpy as np
import re
import os

print("=" * 70)
print("STEP 04 - FEATURE ENGINEERING")
print("=" * 70)


# ==========================================================
# 1. LOAD PREPROCESSED DATA
# ==========================================================

input_file = "output/preprocessed_news.csv"
output_file = "output/featured_news.csv"

df = pd.read_csv(input_file)

print("\nInput Shape:", df.shape)

print("\nInput Columns:")
print(df.columns.tolist())


# ==========================================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "URL",
    "Title",
    "Category",
    "Processed_Title"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing from dataset."
    )


# ==========================================================
# 3. CLEAN TEXT COLUMNS
# ==========================================================

df["Title"] = (
    df["Title"]
    .fillna("")
    .astype(str)
)

df["Processed_Title"] = (
    df["Processed_Title"]
    .fillna("")
    .astype(str)
)

df["URL"] = (
    df["URL"]
    .fillna("")
    .astype(str)
)


# ==========================================================
# 4. BASIC FEATURES
# EXACTLY 7 FEATURES
# ==========================================================

def create_basic_features(row):

    title = row["Title"]
    text = row["Processed_Title"]
    url = row["URL"]

    words = text.split()

    word_count = len(words)

    if word_count > 0:

        avg_word_length = (
            sum(len(word) for word in words)
            / word_count
        )

    else:

        avg_word_length = 0


    # 1. Title Length
    title_length = len(title)


    # 2. Word Count
    # Based on processed title
    word_count = len(words)


    # 3. Average Word Length
    avg_word_length = avg_word_length


    # 4. Digit Count
    digit_count = sum(
        character.isdigit()
        for character in title
    )


    # 5. Special Character Count
    special_character_count = len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            title
        )
    )


    # 6. Question Count
    question_count = title.count("?")


    # 7. URL Length
    url_length = len(url)


    return pd.Series({

        "Title_Length": title_length,

        "Word_Count": word_count,

        "Avg_Word_Length": avg_word_length,

        "Digit_Count": digit_count,

        "Special_Character_Count":
            special_character_count,

        "Question_Count":
            question_count,

        "URL_Length":
            url_length
    })


print("\nCreating basic features...")


basic_features = df.apply(
    create_basic_features,
    axis=1
)


# IMPORTANT:
# Assign columns individually.
# This prevents duplicate-column problems.

for column in basic_features.columns:

    df[column] = basic_features[column]


print("Basic features completed.")


# ==========================================================
# 5. CATEGORY-SPECIFIC KEYWORDS
# ==========================================================

business_keywords = [

    "business",
    "company",
    "companies",
    "profit",
    "profits",
    "revenue",
    "investment",
    "investor",
    "investors",
    "finance",
    "financial",
    "bank",
    "banking",
    "market",
    "stock",
    "stocks",
    "economy",
    "economic"
]


technology_keywords = [

    "technology",
    "technology",
    "tech",
    "ai",
    "artificial intelligence",
    "machine learning",
    "software",
    "hardware",
    "computer",
    "computing",
    "robot",
    "robotics",
    "cloud",
    "cybersecurity",
    "cyber",
    "internet",
    "digital",
    "chip",
    "chips",
    "semiconductor",
    "algorithm",
    "app",
    "apps"
]


politics_keywords = [

    "politics",
    "political",
    "government",
    "president",
    "presidential",
    "prime minister",
    "minister",
    "parliament",
    "senate",
    "congress",
    "election",
    "elections",
    "vote",
    "voting",
    "campaign",
    "politician",
    "politicians",
    "policy",
    "law",
    "legislation"
]


health_keywords = [

    "health",
    "healthcare",
    "medical",
    "medicine",
    "doctor",
    "doctors",
    "hospital",
    "hospitals",
    "patient",
    "patients",
    "disease",
    "diseases",
    "cancer",
    "virus",
    "vaccine",
    "vaccination",
    "treatment",
    "therapy",
    "drug",
    "clinical",
    "symptom",
    "infection",
    "pandemic"
]


energy_keywords = [

    "energy",
    "power",
    "electricity",
    "electric",
    "oil",
    "gas",
    "natural gas",
    "petroleum",
    "fuel",
    "solar",
    "solar power",
    "wind",
    "wind power",
    "renewable",
    "renewable energy",
    "nuclear",
    "nuclear power",
    "coal",
    "battery",
    "batteries",
    "electric vehicle",
    "ev",
    "grid"
]


# ==========================================================
# 6. KEYWORD COUNT FUNCTION
# ==========================================================

def count_keywords(text, keywords):

    text = str(text).lower()

    total_count = 0

    for keyword in keywords:

        pattern = (
            r"\b"
            + re.escape(keyword.lower())
            + r"\b"
        )

        matches = re.findall(
            pattern,
            text
        )

        total_count += len(matches)

    return total_count


# ==========================================================
# 7. CREATE KEYWORD FEATURES
# ==========================================================

print("\nCreating keyword features...")


df["Business_Keyword_Count"] = (
    df["Processed_Title"]
    .apply(
        lambda text:
        count_keywords(
            text,
            business_keywords
        )
    )
)


df["Technology_Keyword_Count"] = (
    df["Processed_Title"]
    .apply(
        lambda text:
        count_keywords(
            text,
            technology_keywords
        )
    )
)


df["Politics_Keyword_Count"] = (
    df["Processed_Title"]
    .apply(
        lambda text:
        count_keywords(
            text,
            politics_keywords
        )
    )
)


df["Health_Keyword_Count"] = (
    df["Processed_Title"]
    .apply(
        lambda text:
        count_keywords(
            text,
            health_keywords
        )
    )
)


df["Energy_Keyword_Count"] = (
    df["Processed_Title"]
    .apply(
        lambda text:
        count_keywords(
            text,
            energy_keywords
        )
    )
)


print("Keyword features completed.")


# ==========================================================
# 8. IMPORTANT ORGANIZATIONS
# ==========================================================

organizations = [

    "Google",
    "Apple",
    "Microsoft",
    "Amazon",
    "Meta",
    "Facebook",
    "Tesla",
    "Nvidia",
    "Intel",
    "IBM",
    "Samsung",
    "Sony",
    "Netflix",
    "OpenAI",
    "SpaceX",
    "Oracle",
    "Adobe",
    "Reuters",
    "BBC",
    "CNN",
    "Bloomberg",
    "Forbes"
]


# ==========================================================
# 9. IMPORTANT PERSONALITIES
# ==========================================================

personalities = [

    "Donald Trump",
    "Joe Biden",
    "Barack Obama",
    "Kamala Harris",
    "Elon Musk",
    "Tim Cook",
    "Bill Gates",
    "Mark Zuckerberg",
    "Sam Altman",
    "Sundar Pichai",
    "Jensen Huang",
    "Jeff Bezos",
    "Xi Jinping",
    "Vladimir Putin",
    "Narendra Modi",
    "Emmanuel Macron",
    "Benjamin Netanyahu"
]


# ==========================================================
# 10. IMPORTANT LOCATIONS
# ==========================================================

locations = [

    "United States",
    "USA",
    "US",
    "United Kingdom",
    "UK",
    "China",
    "India",
    "Pakistan",
    "Russia",
    "Ukraine",
    "France",
    "Germany",
    "Japan",
    "Canada",
    "Australia",
    "Israel",
    "Iran",
    "Iraq",
    "Saudi Arabia",
    "London",
    "New York",
    "Washington"
]


# ==========================================================
# 11. ENTITY COUNT FUNCTION
# ==========================================================

def count_entities(text, entities):

    text = str(text).lower()

    total_count = 0

    for entity in entities:

        pattern = (
            r"\b"
            + re.escape(entity.lower())
            + r"\b"
        )

        matches = re.findall(
            pattern,
            text
        )

        total_count += len(matches)

    return total_count


# ==========================================================
# 12. ORGANIZATION COUNT
# ==========================================================

print("\nCreating organization feature...")


df["Organization_Count"] = (
    df["Title"]
    .apply(
        lambda text:
        count_entities(
            text,
            organizations
        )
    )
)


# ==========================================================
# 13. PERSON COUNT
# ==========================================================

print("Creating personality feature...")


df["Person_Count"] = (
    df["Title"]
    .apply(
        lambda text:
        count_entities(
            text,
            personalities
        )
    )
)


# ==========================================================
# 14. LOCATION COUNT
# ==========================================================

print("Creating location feature...")


df["Location_Count"] = (
    df["Title"]
    .apply(
        lambda text:
        count_entities(
            text,
            locations
        )
    )
)


# ==========================================================
# 15. EXACT 15 FEATURES
# ==========================================================

feature_columns = [

    "Title_Length",

    "Word_Count",

    "Avg_Word_Length",

    "Digit_Count",

    "Special_Character_Count",

    "Question_Count",

    "URL_Length",

    "Business_Keyword_Count",

    "Technology_Keyword_Count",

    "Politics_Keyword_Count",

    "Health_Keyword_Count",

    "Energy_Keyword_Count",

    "Organization_Count",

    "Person_Count",

    "Location_Count"
]


# ==========================================================
# 16. FEATURE VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)


print(
    "\nNumber of features:",
    len(feature_columns)
)


print("\nFeatures:")

for number, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{number:02d}. {feature}"
    )


# Check duplicate columns
duplicate_columns = df.columns[
    df.columns.duplicated()
].tolist()


if duplicate_columns:

    print(
        "\nWARNING - Duplicate columns found:"
    )

    print(
        duplicate_columns
    )

    # Keep first occurrence
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]


# ==========================================================
# 17. NUMERIC CLEANING
# ==========================================================

print("\nCleaning feature values...")


for feature in feature_columns:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    df[feature] = (
        df[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


print("Feature cleaning completed.")


# ==========================================================
# 18. CHECK TARGET
# ==========================================================

print("\nTarget column:")
print("Category")


print("\nCategory Distribution:")

print(
    df["Category"]
    .value_counts()
)


# ==========================================================
# 19. SAVE FEATURED DATASET
# ==========================================================

os.makedirs(
    "output",
    exist_ok=True
)


df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# ==========================================================
# 20. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nOriginal Rows:",
    len(df)
)


print(
    "Final Dataset Shape:",
    df.shape
)


print(
    "Total Engineered Features:",
    len(feature_columns)
)


print(
    "\nSaved File:"
)


print(
    output_file
)


print(
    "\nFirst 5 Rows of 15 Features:"
)


print(
    df[feature_columns]
    .head()
)


print("\n" + "=" * 70)
print("STEP 04 COMPLETED")
print("=" * 70)