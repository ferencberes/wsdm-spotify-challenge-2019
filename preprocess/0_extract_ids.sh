# needs to be run in the correct folder

for i in log_*; do { cat $i | tail -n+2 | cut -d, -f1 | uniq; }; done > ../unique_session_ids.csv
for i in log_prehistory*; do { cat $i | tail -n+2 | cut -d, -f1 | uniq; }; done >> ../unique_session_ids.csv
for i in tf_00000000000*; do { cat $i | tail -n+1 | cut -d, -f1; }; done > ../unique_track_ids.csv