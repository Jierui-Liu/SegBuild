# save_dir=../exp/result_ensemble_highscore
# # python ensemble_filter.py -save_dir $save_dir
# python ensemble.py -save_dir $save_dir

# zip result_ensemble_highscore.zip ../exp/result_ensemble_highscore/*
# mv -f result_ensemble_highscore.zip ../exp

# save_dir=../exp/result_ensemble_highrecall
# # python ensemble_filter.py -save_dir $save_dir
# python ensemble.py -save_dir $save_dir

# zip result_ensemble_highrecall.zip ../exp/result_ensemble_highrecall/*
# mv -f result_ensemble_highrecall.zip ../exp

save_dir=../exp/result_ensemble
python ensemble_filter.py -save_dir $save_dir

zip result_ensemble.zip ../exp/result_ensemble/*
mv -f result_ensemble.zip ../exp