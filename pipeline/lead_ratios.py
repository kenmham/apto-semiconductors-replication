from pathlib import Path
import polars as pl

OUT = Path(__file__).resolve().parents[1] / 'outputs/data'
excluded = {('currentRatingAmps', 'RF FETs, MOSFETs'), ('operatingSupplyCurrent', 'LDO Voltage Regulators')}
leads = pl.read_csv(OUT / 'all_capability_leads.csv')
retained = leads.filter(~pl.struct('column', 'subcategory').map_elements(
    lambda row: (row['column'], row['subcategory']) in excluded, return_dtype=pl.Boolean))
assert retained.height == 204
retained.write_csv(OUT / 'lead_ratios.csv')
