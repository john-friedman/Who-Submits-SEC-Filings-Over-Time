import polars as pl
import matplotlib.pyplot as plt
import numpy as np

print("Reading parquet and extracting filer info...")
df = pl.read_parquet("sec_master_submissions.parquet")

# Deduplicate by accession number, extract filer_cik and year
df_filers = (
    df.select([
        "accessionNumber",
    ])
    .unique(subset=["accessionNumber"])
    .with_columns([
        # Extract filer CIK (first 10 digits)
        pl.col("accessionNumber").str.slice(0, 10).alias("filer_cik"),
        # Extract year from accession number (convert to 4-digit)
        pl.when(pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32) <= 49)
        .then(2000 + pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32))
        .otherwise(1900 + pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32))
        .alias("year")
    ])
)

# Filter to 1994 onwards
df_filers = df_filers.filter(pl.col("year") >= 1994)

print(f"Total unique filings (1994+): {len(df_filers):,}")

print("\nCounting filings per filer per year...")
# Count filings by filer and year
yearly_counts = (
    df_filers.group_by(["year", "filer_cik"])
    .agg(pl.len().alias("filing_count"))
)

# Rank filers within each year
yearly_counts = yearly_counts.with_columns([
    pl.col("filing_count")
    .rank(method="ordinal", descending=True)
    .over("year")
    .alias("rank")
])

# Assign buckets
yearly_counts = yearly_counts.with_columns([
    pl.when(pl.col("rank") <= 10)
    .then(pl.lit("Top 10"))
    .when(pl.col("rank") <= 50)
    .then(pl.lit("Top 11-50"))
    .when(pl.col("rank") <= 100)
    .then(pl.lit("Top 51-100"))
    .otherwise(pl.lit("Other"))
    .alias("bucket")
])

print("\n=== PLOT 1: Market Concentration Over Time ===")
# Calculate bucket shares per year
bucket_totals = (
    yearly_counts.group_by(["year", "bucket"])
    .agg(pl.col("filing_count").sum().alias("bucket_total"))
)

year_totals = (
    yearly_counts.group_by("year")
    .agg(pl.col("filing_count").sum().alias("year_total"))
)

bucket_shares = (
    bucket_totals.join(year_totals, on="year")
    .with_columns([
        (pl.col("bucket_total") / pl.col("year_total") * 100).alias("share_pct")
    ])
    .sort(["year", "bucket"])
)

# Pivot for plotting
pivot_data = bucket_shares.pivot(
    index="year",
    columns="bucket",
    values="share_pct"
).sort("year").fill_null(0)

pivot_pd = pivot_data.to_pandas()
years = pivot_pd["year"].values

# Order buckets for stacking: Top 10 at bottom, then 11-50, then 51-100, Other at top
bucket_order = ["Top 10", "Top 11-50", "Top 51-100", "Other"]
data_arrays = [pivot_pd[col].values if col in pivot_pd.columns else np.zeros(len(years)) for col in bucket_order]

# Create stacked area chart
fig, ax = plt.subplots(figsize=(14, 8))

# Colors: blue at bottom (Top 10), red at top (Other)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

ax.stackplot(years, *data_arrays, labels=bucket_order, colors=colors, alpha=0.8)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Percentage of Total SEC Filings", fontsize=12)
ax.set_title("Who Submits SEC Filings?\n(Submitter CIK = First 10 Digits of Accession Number)", 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)
ax.set_xlim(years.min(), years.max())

plt.tight_layout()
plt.savefig("market_concentration.png", dpi=300, bbox_inches='tight')
print("Saved: market_concentration.png")
plt.close()

print("\n=== PLOT 2: Top 10 Filers Over Time ===")
# Get top 10 for each year
top10_by_year = (
    yearly_counts.filter(pl.col("rank") <= 10)
    .select(["year", "filer_cik", "rank", "filing_count"])
    .sort(["year", "rank"])
)

# Get all unique filers that were ever in top 10
all_top10_filers = top10_by_year.select("filer_cik").unique().sort("filer_cik")
print(f"\nTotal unique filers that were in top 10 at some point: {len(all_top10_filers)}")

# Create complete grid (all years × all top10 filers)
years_list = yearly_counts.select("year").unique().sort("year").to_series().to_list()
filers_list = all_top10_filers.to_series().to_list()

# Pivot to create matrix (filers × years)
rank_matrix = top10_by_year.pivot(
    index="filer_cik",
    columns="year",
    values="rank"
).fill_null(11)  # 11 means "not in top 10"

rank_pd = rank_matrix.to_pandas()
rank_values = rank_pd.drop(columns=["filer_cik"]).values
filers = rank_pd["filer_cik"].values
years_cols = [col for col in rank_pd.columns if col != "filer_cik"]

# Create heatmap
fig, ax = plt.subplots(figsize=(18, 12))

# Use custom colormap: ranks 1-10 are colored, 11 (not in top 10) is gray
cmap = plt.cm.RdYlGn_r.copy()
cmap.set_over('lightgray')

im = ax.imshow(rank_values, aspect='auto', cmap=cmap, vmin=1, vmax=10, interpolation='nearest')

# Set ticks
ax.set_xticks(np.arange(len(years_cols)))
ax.set_yticks(np.arange(len(filers)))
ax.set_xticklabels(years_cols, rotation=45, ha='right')
ax.set_yticklabels(filers, fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, extend='max')
cbar.set_label("Rank (1=Most Filings, Gray=Not in Top 10)", fontsize=11)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Submitter CIK (First 10 Digits of Accession Number)", fontsize=12)
ax.set_title("Top 10 Filing Submitters: Composition and Rankings Over Time", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("top10_composition_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved: top10_composition_heatmap.png")
plt.close()

print("\n=== Summary ===")
print(f"Years analyzed: {min(years_list)} - {max(years_list)}")
print(f"Unique filers overall: {df_filers['filer_cik'].n_unique():,}")
print(f"Unique filers ever in top 10: {len(all_top10_filers):,}")