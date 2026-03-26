"""
apriori_model.py
================
Discovers association rules between products using a simplified Apriori approach.

How it works (plain English):
  1. Read every transaction (basket) from the CSV.
  2. Count how often each item and each pair of items appear.
  3. If a pair appears often enough (min_support), check how reliable
     the rule "A → B" is (confidence = how often B appears given A).
  4. Store rules with lift > 1 (lift means the pair appears more than by chance).
  5. Expose helpers so Flask can ask "what goes with Milk?" or "give me bundles".
"""

import pandas as pd
from itertools import combinations
from collections import defaultdict


class AprioriModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.rules     = pd.DataFrame()   # DataFrame of (antecedent, consequent, support, confidence, lift)
        self.all_items = set()            # Every unique item seen in the data
        self._run()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_transactions(self) -> list:
        """Read CSV and return a list of frozensets (one per row)."""
        df = pd.read_csv(self.data_path)
        col = {c.lower(): c for c in df.columns}.get("items", df.columns[0])
        return df[col].apply(
            lambda x: frozenset(
                i.strip() for i in str(x).split(",")
                if i.strip() and i.strip().lower() != "nan"
            )
        ).tolist()

    def _get_thresholds(self, n: int):
        """Use looser thresholds for large datasets so we still get useful rules."""
        if n > 5000:
            return 0.005, 0.05   # 0.5% support, 5% confidence
        return 0.01, 0.10        # 1% support, 10% confidence

    def _run(self):
        """Full pipeline: load → count → filter → generate rules."""
        try:
            transactions = self._load_transactions()
            n = len(transactions)
            min_sup, min_conf = self._get_thresholds(n)

            # Step 1 – single-item counts
            item_counts: dict[frozenset, int] = defaultdict(int)
            for t in transactions:
                for item in t:
                    item_counts[frozenset([item])] += 1
                    self.all_items.add(item)

            # Step 2 – pair counts
            pair_counts: dict[frozenset, int] = defaultdict(int)
            for t in transactions:
                for pair in combinations(sorted(t), 2):
                    pair_counts[frozenset(pair)] += 1

            # Step 3 – build rules from frequent pairs
            rules_list = []
            for pair, pair_count in pair_counts.items():
                support_ab = pair_count / n
                if support_ab < min_sup:
                    continue

                a, b = list(pair)
                sup_a = item_counts[frozenset([a])] / n
                sup_b = item_counts[frozenset([b])] / n
                lift  = support_ab / (sup_a * sup_b) if (sup_a * sup_b) > 0 else 0

                # Rule A → B
                conf_a = support_ab / sup_a if sup_a > 0 else 0
                if conf_a >= min_conf:
                    rules_list.append({
                        "antecedents": frozenset([a]),
                        "consequents": frozenset([b]),
                        "support":     support_ab,
                        "confidence":  conf_a,
                        "lift":        lift,
                    })

                # Rule B → A
                conf_b = support_ab / sup_b if sup_b > 0 else 0
                if conf_b >= min_conf:
                    rules_list.append({
                        "antecedents": frozenset([b]),
                        "consequents": frozenset([a]),
                        "support":     support_ab,
                        "confidence":  conf_b,
                        "lift":        lift,
                    })

            self.rules = (
                pd.DataFrame(rules_list).sort_values("lift", ascending=False)
                if rules_list else pd.DataFrame()
            )

        except Exception as e:
            print(f"[Apriori] Error during analysis: {e}")
            self.rules = pd.DataFrame()

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(self, item: str) -> list:
        """Return items that are frequently bought with `item`."""
        if self.rules.empty:
            return []
        item = item.strip()
        result = set()
        for _, row in self.rules.iterrows():
            if item in row["antecedents"]:
                result.update(row["consequents"])
        return sorted(result)

    def get_bundles(self, n: int = 5) -> list:
        """Return top-n unique product bundles (pairs with highest lift)."""
        if self.rules.empty:
            return []
        seen, bundles = set(), []
        for _, row in self.rules.iterrows():
            bundle = tuple(sorted(row["antecedents"] | row["consequents"]))
            if bundle not in seen:
                seen.add(bundle)
                bundles.append(list(bundle))
            if len(bundles) >= n:
                break
        return bundles

    def get_top_rules_chart_data(self, top_n: int = 8) -> dict:
        """Return label / confidence / lift lists for the chart."""
        if self.rules.empty:
            return {"labels": [], "confidence": [], "lift": []}
        top = self.rules.head(top_n)
        labels = [
            f"{list(r.antecedents)[0][:10]}→{list(r.consequents)[0][:10]}"
            for _, r in top.iterrows()
        ]
        return {
            "labels":     labels,
            "confidence": [round(float(c), 3) for c in top["confidence"]],
            "lift":       [round(float(l), 3) for l in top["lift"]],
        }