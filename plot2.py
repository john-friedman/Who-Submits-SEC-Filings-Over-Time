import polars as pl
import matplotlib.pyplot as plt
import numpy as np

print("Reading parquet and extracting submitter and company info...")
df = pl.read_parquet("sec_master_submissions.parquet")

# Extract submitter CIK, company CIK, and year
df_submitters = (
    df.select([
        "accessionNumber",
        "cik",
    ])
    .unique(subset=["accessionNumber"])
    .with_columns([
        # Extract submitter CIK (first 10 digits)
        pl.col("accessionNumber").str.slice(0, 10).alias("submitter_cik"),
        # Extract year from accession number
        pl.when(pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32) <= 49)
        .then(2000 + pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32))
        .otherwise(1900 + pl.col("accessionNumber").str.slice(11, 2).cast(pl.Int32))
        .alias("year")
    ])
    .filter(pl.col("year") >= 1994)
    .select(["submitter_cik", "cik", "year"])
)

print(f"Total unique filings (1994+): {len(df_submitters):,}")

print("\nIdentifying top 10 submitters overall...")
# Get top 10 submitters by total number of filings
top10_submitters = (
    df_submitters.group_by("submitter_cik")
    .agg(pl.len().alias("total_filings"))
    .sort("total_filings", descending=True)
    .head(10)
    .select("submitter_cik")
    .to_series()
    .to_list()
)

print(f"Top 10 submitters: {top10_submitters}")

print("\nCounting unique companies per submitter per year...")
# For each (submitter, year), count unique companies
companies_per_year = (
    df_submitters.filter(pl.col("submitter_cik").is_in(top10_submitters))
    .group_by(["submitter_cik", "year"])
    .agg(pl.col("cik").n_unique().alias("num_companies"))
    .sort(["submitter_cik", "year"])
)

# Convert to pandas for plotting
companies_pd = companies_per_year.to_pandas()

print("\n=== Creating Facet Grid Plot (Log Scale) ===")
# Calculate global min/max for y-axis
y_min = companies_pd["num_companies"].min()
y_max = companies_pd["num_companies"].max()

# Create 2x5 grid for 10 submitters
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

for idx, submitter in enumerate(top10_submitters):
    ax = axes[idx]
    
    # Get data for this submitter
    submitter_data = companies_pd[companies_pd["submitter_cik"] == submitter]
    
    # Plot line
    ax.plot(submitter_data["year"], submitter_data["num_companies"], 
            marker='o', linewidth=2, markersize=3, color='steelblue')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Customize subplot
    ax.set_title(f"CIK: {submitter}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1994, companies_pd["year"].max())
    
    # Add y-axis label only for leftmost plots
    if idx % 5 == 0:
        ax.set_ylabel("Unique Companies (log scale)", fontsize=9)
    
    # Add x-axis label only for bottom plots
    if idx >= 5:
        ax.set_xlabel("Year", fontsize=9)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

# Overall title
fig.suptitle("Number of Unique Companies Filed For by Top 10 Submitters Over Time (Log Scale)", 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("companies_per_submitter_facets.png", dpi=300, bbox_inches='tight')
print("Saved: companies_per_submitter_facets.png")
plt.close()

print("\n=== Summary Statistics ===")
for submitter in top10_submitters:
    submitter_data = companies_pd[companies_pd["submitter_cik"] == submitter]
    print(f"\nSubmitter {submitter}:")
    print(f"  Years active: {submitter_data['year'].min()} - {submitter_data['year'].max()}")
    print(f"  Max companies: {submitter_data['num_companies'].max():,}")
    print(f"  Min companies: {submitter_data['num_companies'].min():,}")
    print(f"  Latest (most recent year): {submitter_data['num_companies'].iloc[-1]:,}")