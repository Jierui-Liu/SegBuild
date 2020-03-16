

save_dir=../exp/result_edge
python edge_noise.py -save_dir $save_dir

zip result_edge.zip ../exp/result_edge/*
mv -f result_edge.zip ../exp