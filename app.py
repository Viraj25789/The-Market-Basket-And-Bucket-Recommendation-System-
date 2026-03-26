"""
app.py
======
Flask application entry point.

Route overview:
  GET  /            → Render the main dashboard
  POST /            → Handle item search or user lookup (same page)
  POST /upload      → Replace the active CSV dataset
  GET  /download_report → Generate and download a plain-text report
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

from model.apriori_model         import AprioriModel
from model.collaborative_filtering import CollaborativeFiltering

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Active dataset path (changes when user uploads a new CSV)
DATA_PATH = os.path.join(UPLOAD_FOLDER, "default.csv")

# ML models (re-created on each new upload)
apriori_model = AprioriModel(DATA_PATH)
cf_model      = CollaborativeFiltering(DATA_PATH)


# ── Utility functions ──────────────────────────────────────────────────────────

def get_item_frequency(path: str, top_n: int = 15) -> tuple:
    """Return (labels, counts) for the top-n most purchased items."""
    try:
        df  = pd.read_csv(path)
        col = {c.lower(): c for c in df.columns}.get("items", df.columns[0])
        flat = [
            i.strip()
            for sublist in df[col].apply(lambda x: str(x).split(","))
            for i in sublist
            if i.strip() and i.strip().lower() != "nan"
        ]
        counts = pd.Series(flat).value_counts().head(top_n)
        return counts.index.tolist(), [int(v) for v in counts.values]
    except Exception:
        return [], []


def get_low_performers(path: str, limit: int = 5) -> list:
    """Return items with the lowest purchase frequency."""
    try:
        df  = pd.read_csv(path)
        col = {c.lower(): c for c in df.columns}.get("items", df.columns[0])
        flat = [
            i.strip()
            for sublist in df[col].apply(lambda x: str(x).split(","))
            for i in sublist
            if i.strip() and i.strip().lower() != "nan"
        ]
        return pd.Series(flat).value_counts(ascending=True).head(limit).index.tolist()
    except Exception:
        return []


def get_summary_stats(path: str) -> dict:
    """Return a dict of KPI values for the overview section."""
    try:
        df      = pd.read_csv(path)
        col_map = {c.lower(): c for c in df.columns}
        items_col = col_map.get("items", df.columns[0])
        user_col  = col_map.get("userid") or col_map.get("user_id")

        total_tx  = len(df)
        total_usr = df[user_col].nunique() if user_col and user_col in df.columns else df.iloc[:, 0].nunique()
        baskets   = df[items_col].apply(lambda x: [i.strip() for i in str(x).split(",")])
        avg_size  = round(baskets.apply(len).mean(), 1)

        return {
            "db_name":      os.path.basename(path),
            "total_tx":     total_tx,
            "total_users":  total_usr,
            "avg_basket":   avg_size,
        }
    except Exception:
        return {"db_name": "Error", "total_tx": 0, "total_users": 0, "avg_basket": 0}


def get_user_purchase_stats(user_id: str, cf) -> dict:
    """
    Calculate real purchase stats for a user directly from the user-item matrix.

    Returns:
      items_bought   : how many unique items this user has purchased
      total_items    : total unique items in the catalogue
      coverage_pct   : items_bought / total_items * 100  (0–100)
      avg_items      : average items bought across ALL users
      percentile     : what % of users this user buys MORE than
    """
    empty = {"items_bought": 0, "total_items": 0, "coverage_pct": 0,
             "avg_items": 0, "percentile": 0}

    if cf.user_item_matrix is None:
        return empty

    matrix = cf.user_item_matrix

    # Resolve user ID (might be int or str in the index)
    uid = None
    if user_id in matrix.index:
        uid = user_id
    else:
        try:
            if int(user_id) in matrix.index:
                uid = int(user_id)
        except (ValueError, TypeError):
            pass

    total_items = len(matrix.columns)

    if uid is None:
        # Unknown user — still return catalogue size so UI can show it
        return {**empty, "total_items": total_items}

    # Per-user purchase counts (number of 1s per row)
    per_user_counts = matrix.sum(axis=1)
    items_bought    = int(per_user_counts[uid])
    avg_items       = round(per_user_counts.mean(), 1)

    # Loyalty score = percentage of the catalogue this user has bought
    # This is honest: if they bought 8 out of 80 items → score = 10
    coverage_pct = round((items_bought / total_items) * 100, 1) if total_items > 0 else 0

    # Percentile: what share of users buy FEWER items than this user
    percentile = round((per_user_counts < items_bought).sum() / len(per_user_counts) * 100)

    return {
        "items_bought": items_bought,
        "total_items":  total_items,
        "coverage_pct": coverage_pct,
        "avg_items":    avg_items,
        "percentile":   percentile,
    }


def build_user_profile(user_id: str, cf) -> dict:
    """
    Build a user profile using real purchase data — not recommendation count.

    Loyalty score  = % of catalogue items this user has actually purchased.
    Churn risk     = based on whether they buy more or less than the average user.
    Status         = based on their purchase percentile among all users.
    """
    recs   = cf.recommend_for_user(user_id)
    is_new = not recs            # True if CF found no similar users (no history)

    if is_new:
        recs = cf.get_popular_items()   # Fall back to popular items

    stats = get_user_purchase_stats(user_id, cf)

    items_bought = stats["items_bought"]
    avg_items    = stats["avg_items"]
    percentile   = stats["percentile"]
    loyalty      = stats["coverage_pct"]   # 0–100, based on actual catalogue coverage

    # Status: top 25% of buyers → Elite, otherwise Regular, unknown → New
    if is_new or items_bought == 0:
        status = "New / Inactive"
    elif percentile >= 75:
        status = "Elite Buyer"
    elif percentile >= 40:
        status = "Regular Buyer"
    else:
        status = "Occasional Buyer"

    # Churn risk: buying significantly below average is a warning sign
    if is_new or items_bought == 0:
        churn_risk = "High"
    elif items_bought >= avg_items * 0.8:
        churn_risk = "Low"
    elif items_bought >= avg_items * 0.4:
        churn_risk = "Medium"
    else:
        churn_risk = "High"

    return {
        "recommendations": recs,
        "is_new_user":     is_new,
        "status":          status,
        "churn_risk":      churn_risk,
        "loyalty_score":   loyalty,         # Real % of catalogue purchased
        "items_bought":    items_bought,
        "total_items":     stats["total_items"],
        "avg_items":       avg_items,
        "percentile":      percentile,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """Accept a new CSV, save it, and re-initialise both models."""
    global apriori_model, cf_model, DATA_PATH

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    filename    = secure_filename(file.filename)
    new_path    = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(new_path)

    DATA_PATH    = new_path
    apriori_model = AprioriModel(DATA_PATH)
    cf_model      = CollaborativeFiltering(DATA_PATH)

    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    # Form values
    user_id = request.form.get("user_id", "Guest").strip() or "Guest"
    item    = request.form.get("item", "").strip()

    # Product search (Apriori)
    item_recs = apriori_model.recommend(item) if item else []

    # User profile (Collaborative Filtering)
    profile = build_user_profile(user_id, cf_model)

    # Dataset stats
    stats         = get_summary_stats(DATA_PATH)
    low_performers = get_low_performers(DATA_PATH)
    items, counts  = get_item_frequency(DATA_PATH)

    # Chart data for association rules
    chart = apriori_model.get_top_rules_chart_data()

    return render_template(
        "index.html",
        # User
        user_id            = user_id,
        user_recommendations = profile["recommendations"],
        user_behavior      = profile,
        # Product search
        item               = item,
        item_recommendations = item_recs,
        # Bundles & chart
        bundles            = apriori_model.get_bundles(),
        chart_labels       = chart["labels"],
        chart_confidence   = chart["confidence"],
        chart_lift         = chart["lift"],
        # Item frequency chart
        items  = items,
        counts = counts,
        # KPIs
        active_db_name     = stats["db_name"],
        total_transactions = stats["total_tx"],
        total_users        = stats["total_users"],
        avg_basket_size    = stats["avg_basket"],
        low_performers     = low_performers,
        total_active_rules = len(apriori_model.rules),
        total_tracked_items = len(cf_model.all_items),
    )


@app.route("/download_report")
def download_report():
    """Generate and return a plain-text executive report as a download."""
    user_id = request.args.get("user_id", "Guest")
    stats   = get_summary_stats(DATA_PATH)
    profile = build_user_profile(user_id, cf_model)
    bundles = apriori_model.get_bundles(5)

    bundle_lines = "\n".join(
        f"  Bundle {i}: {' + '.join(b)}"
        for i, b in enumerate(bundles, 1)
    ) or "  No bundles detected."

    report = f"""
==============================================================
      MARKET BASKET & BUNDLING ENGINE — EXECUTIVE REPORT
==============================================================
Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset   : {stats['db_name']}
User      : #{user_id}

[PART 1 — MARKET OVERVIEW]
--------------------------------------------------------------
Total Transactions  : {stats['total_tx']}
Unique Users        : {stats['total_users']}
Avg. Basket Size    : {stats['avg_basket']}
Unique Products     : {len(cf_model.all_items)}

[PART 2 — INVENTORY ALERTS]
--------------------------------------------------------------
Low-Performing Items:
  {', '.join(get_low_performers(DATA_PATH)) or 'None detected.'}

[PART 3 — ASSOCIATION & BUNDLING]
--------------------------------------------------------------
Association Rules Found : {len(apriori_model.rules)}
Top Product Bundles:
{bundle_lines}

[PART 4 — USER BEHAVIOUR]
--------------------------------------------------------------
User ID              : {user_id}
Engagement Rank      : {profile['status']}
Churn Risk           : {profile['churn_risk']}
Items Purchased      : {profile['items_bought']} of {profile['total_items']} catalogue items
Catalogue Coverage   : {profile['loyalty_score']}%
Buyer Percentile     : Top {100 - profile['percentile']}% (buys more than {profile['percentile']}% of users)
Avg. User Buys       : {profile['avg_items']} items
Recommended Items    : {', '.join(profile['recommendations']) or 'No history found.'}

==============================================================
                       END OF REPORT
==============================================================
"""

    return Response(
        report,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=Report_{user_id}.txt"},
    )


if __name__ == "__main__":
    app.run(debug=True)