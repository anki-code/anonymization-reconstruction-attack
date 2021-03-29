# Reconstruction attack on anonymized dataset

## Data:
We used the MIMIC-III dataset which can be obtained at [MIMIC-III database](https://mimic.physionet.org/).

## Data Processing:
- To run only_demographic.py and los_prediction.py, we used the data processing from [Tang et al. (2018)](https://github.com/illidanlab/urgent-care-comparative)
- To run reconstruction_attack.py, we used the data processing from [Pabkin et al. (2018)](https://github.com/apakbin94/ICU72hReadmissionMIMICIII)
- For obtaining the anonymized version of each of the dataset, we used the [ARX tool](https://arx.deidentifier.org/)
