echo "Track info compression STARTED"
python compress_track_info.py
echo "Track info compression DONE"
echo "Track statistics calculation STARTED"
python calculate_track_stats.py first
python calculate_track_stats.py second
python calculate_track_stats.py both
echo "Track statistics calculation DONE"
echo "Feature engineering for GBT STARTED"
python feature_engineering.py 2018-07-15 2018-07-22
python feature_engineering.py 2018-07-30 2018-08-05
python feature_engineering.py 2018-08-13 2018-08-19
python feature_engineering.py 2018-08-27 2018-09-02
python feature_engineering.py 2018-09-10 2018-09-16
echo "Feature engineering for GBT DONE"