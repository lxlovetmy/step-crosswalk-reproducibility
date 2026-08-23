# Required public inputs

No raw or participant-level data are included in this package.

Create data/raw inside the package only for a local run, then place the twelve files at the exact relative paths listed in expected_inputs.csv. The expected structure is:

    data/raw/nhanes_2011_2012/DEMO_G.XPT
    data/raw/nhanes_2011_2012/BMX_G.XPT
    data/raw/nhanes_2013_2014/DEMO_H.XPT
    data/raw/nhanes_2013_2014/BMX_H.XPT
    data/raw/physionet_v1.0.1/nhanes_1440_troianowear.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_actisteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_adeptsteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_oaksteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_scrfsteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_scsslsteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_vssteps.csv.xz
    data/raw/physionet_v1.0.1/nhanes_1440_vsrevsteps.csv.xz

The four DEMO/BMX files are NHANES public-use files. The eight compressed minute-level files are from the PhysioNet v1.0.1 source snapshot used by the study; its accompanying source license is CC0 1.0. Always consult the official source pages and terms at the time of download.

Verify SHA-256 values before analysis. A checksum mismatch means the input is not the frozen study input and the pipeline must stop. Do not commit data/raw, derived participant-level tables, fold assignments, out-of-fold participant predictions, or bootstrap replicate records.

Official acquisition pages:

- NHANES 2011–2012 and 2013–2014 public-use DEMO/BMX components: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx
- PhysioNet project snapshot used by the study: obtain version 1.0.1 from its official project page and retain its accompanying CC0 notice.

No automatic downloader is included because source terms and page locations should be reviewed by the person obtaining the files. The pipeline refuses any input whose content differs from `expected_inputs.csv`.
