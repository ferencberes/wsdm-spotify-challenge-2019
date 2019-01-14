DATA_DIR=$1
EXPERIMENT_DIR=$2
MAX_THREADS=20
echo $DATA_DIR
echo $EXPERIMENT_DIR
echo $MAX_THREADS

echo "Track info compression STARTED"
python compress_track_info.py $DATA_DIR $MAX_THREADS
echo "Track info compression DONE"

echo "Track statistics calculation STARTED"
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-15 2018-09-18 9 first
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-15 2018-09-18 9 second
python calculate_track_stats.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-15 2018-09-18 9 both
echo "Track statistics calculation DONE"

echo "Feature engineering for GBT STARTED"
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-15 2018-07-22 3 track_stats_2018-07-15_2018-09-18
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-30 2018-08-05 3 track_stats_2018-07-15_2018-09-18
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-08-13 2018-08-19 3 track_stats_2018-07-15_2018-09-18
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-08-27 2018-09-02 3 track_stats_2018-07-15_2018-09-18
python feature_engineering.py $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-10 2018-09-16 3 track_stats_2018-07-15_2018-09-18
echo "Feature engineering for GBT DONE"

echo "GBT model training STARTED"
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-07-15_2018-07-22_with_track_stats_2018-07-15_2018-09-18 200000
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-07-30_2018-08-05_with_track_stats_2018-07-15_2018-09-18 200000
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-08-13_2018-08-19_with_track_stats_2018-07-15_2018-09-18 200000
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-08-27_2018-09-02_with_track_stats_2018-07-15_2018-09-18 200000
python train_gbt_model.py $EXPERIMENT_DIR $MAX_THREADS 2018-09-10_2018-09-16_with_track_stats_2018-07-15_2018-09-18 200000
echo "GBT model training DONE"

echo "Prediction with GBT STARTED"
bash predict_all.sh $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-15_2018-07-22_with_track_stats_2018-07-15_2018-09-18
bash predict_all.sh $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-07-30_2018-08-05_with_track_stats_2018-07-15_2018-09-18
bash predict_all.sh $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-08-13_2018-08-19_with_track_stats_2018-07-15_2018-09-18
bash predict_all.sh $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-08-27_2018-09-02_with_track_stats_2018-07-15_2018-09-18
bash predict_all.sh $DATA_DIR $EXPERIMENT_DIR $MAX_THREADS 2018-09-10_2018-09-16_with_track_stats_2018-07-15_2018-09-18
echo "Prediction with GBT DONE"