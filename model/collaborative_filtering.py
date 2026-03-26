"""
collaborative_filtering.py
===========================
Recommends products to a user based on what *similar users* have bought.

How it works (plain English):
  1. Build a user-item matrix: rows = users, columns = items, value = 1 if bought.
  2. Calculate cosine similarity between every pair of users.
     (Two users are similar if they bought many of the same things.)
  3. To recommend for User X:
       - Find the most similar users.
       - Collect items those users bought that X has NOT bought.
       - Rank by weighted similarity score.
  4. If the user is unknown, fall back to the most popular items overall.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    def __init__(self, data_path: str):
        self.data_path         = data_path
        self.user_item_matrix  = None   # DataFrame: users × items (0/1)
        self.similarity_matrix = None   # DataFrame: users × users (float)
        self.all_items         = []     # Sorted list of all unique items
        self._prepare()

    # ── Private ────────────────────────────────────────────────────────────────

    def _prepare(self):
        """Load data and build both matrices."""
        try:
            df = pd.read_csv(self.data_path)
            col = {c.lower(): c for c in df.columns}

            items_col = col.get("items")
            user_col  = col.get("userid") or col.get("user_id") or col.get("user")

            # Both columns must exist to build the model
            if not items_col or not user_col:
                print("[CF] Missing 'Items' or 'UserID' column – skipping.")
                return

            # Clean items: split by comma, strip whitespace, drop 'nan'
            df[items_col] = df[items_col].apply(
                lambda x: [
                    i.strip() for i in str(x).split(",")
                    if i.strip() and i.strip().lower() != "nan"
                ]
            )

            self.all_items = sorted({item for row in df[items_col] for item in row})
            users = df[user_col].unique()

            # Build user-item matrix (binary purchase history)
            self.user_item_matrix = pd.DataFrame(0, index=users, columns=self.all_items)
            for _, row in df.iterrows():
                for item in row[items_col]:
                    self.user_item_matrix.at[row[user_col], item] = 1

            # Build cosine similarity matrix
            sim = cosine_similarity(self.user_item_matrix)
            self.similarity_matrix = pd.DataFrame(sim, index=users, columns=users)

        except Exception as e:
            print(f"[CF] Preparation error: {e}")

    def _resolve_user_id(self, user_id):
        """Try matching user_id as-is, then as int, then as str."""
        if self.user_item_matrix is None:
            return None
        idx = self.user_item_matrix.index
        if user_id in idx:
            return user_id
        try:
            as_int = int(user_id)
            if as_int in idx:
                return as_int
        except (ValueError, TypeError):
            pass
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend_for_user(self, user_id, top_n: int = 5) -> list:
        """Return up to top_n item recommendations for the given user."""
        if self.user_item_matrix is None or self.similarity_matrix is None:
            return []

        uid = self._resolve_user_id(user_id)
        if uid is None:
            return []   # Unknown user → caller should fall back to popular items

        # Items this user already bought
        owned = set(self.user_item_matrix.columns[self.user_item_matrix.loc[uid] == 1])

        # Score candidate items by weighted similarity
        scores: dict = {}
        sim_scores = self.similarity_matrix[uid].drop(labels=uid).sort_values(ascending=False)

        for other_uid, sim_score in sim_scores.items():
            if sim_score <= 0:
                continue
            other_owned = set(self.user_item_matrix.columns[self.user_item_matrix.loc[other_uid] == 1])
            for item in (other_owned - owned):
                scores[item] = scores.get(item, 0) + sim_score

        return sorted(scores, key=scores.get, reverse=True)[:top_n]

    def get_popular_items(self, top_n: int = 5) -> list:
        """Return the most purchased items across all users."""
        if self.user_item_matrix is None:
            return []
        return self.user_item_matrix.sum().sort_values(ascending=False).index[:top_n].tolist()