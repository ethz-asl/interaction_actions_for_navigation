rm /tmp/results_loc_live_eval_scrubbed_ckk.txt
rm /tmp/results_loc_live_eval_corridor_koze_kids.txt

for START in {0..430..10}
# for START in {290..720..10} # for demo2
do
  mon launch loc_live_eval.launch rosbag:=corridor_koze_kids start_time:=$START duration:=20 rviz:=false
  mon launch loc_live_eval.launch rosbag:=scrubbed_ckk start_time:=$START duration:=20 rviz:=false
done
