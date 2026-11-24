# Datasets (not included in this repository)

Data files are **not** tracked in git because of their size. The BRFSS source data are
publicly available from the CDC: https://www.cdc.gov/brfss/annual_data/annual_data.htm
(2024 LLCP codebook: `USCODE24_LLCP_082125.HTML`).

The pipelines expect harmonized, pre-split CSVs in `02_datasets/processed/`:

```
02_datasets/processed/
├── brfss_stroke_ny_train.csv
├── brfss_stroke_ny_val.csv
├── brfss_stroke_ny_test.csv
├── brfss_stroke_all_states_train.csv
├── brfss_stroke_all_states_val.csv
├── brfss_stroke_all_states_test.csv
└── pooling_summary_by_year.csv          # used by make_manuscript_figures.py
```

Required metadata columns: `survey_year`, `state_fips`, `split`, `stroke` (0/1 target),
`survey_weight`, `strata`, `psu`, `seqno`. Every other column is treated as a predictor.

Temporal split used in the study: **train 2015–2019, validation 2020–2022, test 2023–2024**.
