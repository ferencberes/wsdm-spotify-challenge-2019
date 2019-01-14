# Final GBT model building

This folder contains our code related to feature engineering and training for the final 5 Gradient Boosted (GBT) models. The majority decision of these models give our final skip prediction.

By running 'run_all.sh' script you can reproduce our results. 

```bash
bash run_all.sh DATA_DIR EXPERIMENT_DIR
```

You have to specify the following 3 parameters:

- **DATA_DIR:** the root folder where your data and pre-generated autoencoder and skip pattern based features reside
- **EXPERIMENT_DIR:** the folder where you want to save your models, and predictions
- **MAX_THREADS:** the maximum number of threads you can allow to run this code (specify this in the 'run_all.sh' script)