path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_24000.pth
config=configs.default_w9_pesudo_highscore_2
python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_32000.pth
config=configs.default_w9_pesudo_highscore_2
python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_40000.pth
config=configs.default_w9_pesudo_highscore_2
python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

zip result_24000_pesudo_highscore_2.zip ../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_24000/*
mv -f result_24000_pesudo_highscore_2.zip ../exp
zip result_32000_pesudo_highscore_2.zip ../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_32000/*
mv -f result_32000_pesudo_highscore_2.zip ../exp
zip result_40000_pesudo_highscore_2.zip ../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_2_40000/*
mv -f result_40000_pesudo_highscore_2.zip ../exp