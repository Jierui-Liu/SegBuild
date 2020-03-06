path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_24000.pth
config=configs.default_w9_pesudo_highscore
python inference.py -config_file $config  -device 1 -path $path # -save_dir $save_dir #-multi-gpu

path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_32000.pth
config=configs.default_w9_pesudo_highscore
python inference.py -config_file $config  -device 1 -path $path # -save_dir $save_dir #-multi-gpu

path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_40000.pth
config=configs.default_w9_pesudo_highscore
python inference.py -config_file $config  -device 1 -path $path # -save_dir $save_dir #-multi-gpu

zip result_42000_pesudo_highscore.zip ../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_24000/*
mv -f result_42000_pesudo_highscore.zip ../exp
zip result_50000_pesudo_highscore.zip ../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_32000/*
mv -f result_50000_pesudo_highscore.zip ../exp
zip result_50000_pesudo_highscore.zip ../exp/20200306_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_40000/*
mv -f result_50000_pesudo_highscore.zip ../exp