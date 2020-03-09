path=../output/model/hrnet_w48_ocr_up4_default_w95_pesudo_highscore_3_28000.pth
config=configs.default_w95_pesudo_highscore_3
python inference.py -config_file $config  -device 3 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_32000.pth
# config=configs.default_w9_pesudo_highscore_3
# python inference.py -config_file $config  -device 3 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_40000.pth
# config=configs.default_w9_pesudo_highscore_3
# python inference.py -config_file $config  -device 3 -path $path # -save_dir $save_dir #-multi-gpu

zip result_28000_pesudo_highscore_3.zip ../exp/20200309_hrnet_w48_ocr_up4_default_w95_pesudo_highscore_3_28000/*
mv -f result_28000_pesudo_highscore_3.zip ../exp
# zip result_32000_pesudo_highscore_3.zip ../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_32000/*
# mv -f result_32000_pesudo_highscore_3.zip ../exp
# zip result_40000_pesudo_highscore_3.zip ../exp/20200308_hrnet_w48_ocr_up4_default_w9_pesudo_highscore_3_40000/*
# mv -f result_40000_pesudo_highscore_3.zip ../exp