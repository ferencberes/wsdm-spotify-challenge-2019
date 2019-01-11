# wsdm-spotify-challenge-2019

In this repository, we present the solution of the team Definitive Turtles for ACM WSDM Cup 2019 [Spotify Sequential Skip Prediction Challenge](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge) that reached 10th place on the final leaderboard.

# Requirements

- Python 3.5 conda environment
- Python packages: turicreate, pandas, numpy, pytorch, scikit-learn
- In our solution we heavily depend on the parallel processing power of the turicreate package.

# Introduction

Our solution is based on gradient boosted trees (GBT) combining several statistical features as well as the output of a recurrent neural network classifier. By using GBT, we were able to assess the importance of the various features and classifiers for skip prediction.

# Usage

Please follow the steps in order if you want to reproduce our final solution:

1. Preprosessing: please follow the instructions [here](preprocess/)
2. Deep learning models:  please follow the instructions [here](methods/)
3. Gradient Boosted Trees: related instructions are coming SOON.

If you have any problem with the code please contact the team members, [Ferenc Béres](mailto:beres@sztaki.hu) or [Domokos Kelen](mailto:kdomokos@sztaki.hu).
