DATA_DIR=$1
EXPERIMENT_DIR=$2
MAX_THREADS=20
echo $DATA_DIR
echo $EXPERIMENT_DIR
echo $MAX_THREADS

echo "Track statistics calculation STARTED"
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-18 2018-09-18 0 first
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-18 2018-09-18 0 second
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-18 2018-09-18 0 both
echo "Track statistics calculation DONE"

echo "Feature engineering for GBT STARTED"
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-18 2018-09-18 0 track_stats_2018-09-18_2018-09-18
echo "Feature engineering for GBT DONE"

echo "GBT model training STARTED"
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-09-18_2018-09-18_with_track_stats_2018-09-18_2018-09-18 10000
echo "GBT model training DONE"

echo "Prediction with GBT STARTED"
python predict_with_model.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-18 2018-09-18 0 2018-09-18_2018-09-18_with_track_stats_2018-09-18_2018-09-18
echo "Prediction with GBT DONE"