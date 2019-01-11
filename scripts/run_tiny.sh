echo "Track statistics calculation STARTED"
python calculate_track_stats.py 2018-09-18 2018-09-18 0 first
python calculate_track_stats.py 2018-09-18 2018-09-18 0 second
python calculate_track_stats.py 2018-09-18 2018-09-18 0 both
echo "Track statistics calculation DONE"
echo "Feature engineering for GBT STARTED"
python feature_engineering.py 2018-09-18 2018-09-18 0 track_stats_2018-09-18_2018-09-18
echo "Feature engineering for GBT DONE"
echo "GBT model training STARTED"
python train_gbt_model.py 2018-09-18_2018-09-18_with_track_stats_2018-09-18_2018-09-18 10000
echo "GBT model training DONE"
echo "Prediction with GBT STARTED"
python predict_with_model.py 2018-09-18 2018-09-18 0 2018-09-18_2018-09-18_with_track_stats_2018-09-18_2018-09-18
echo "Prediction with GBT DONE"