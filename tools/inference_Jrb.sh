path=../output/model/hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_7_32000.pth
config=configs.default_4_pesudo_highscore_2_continue_7
python inference.py -config_file $config  -device 1 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_4_w9_pesudo_highscore_2_30000.pth
# config=configs.default_4_w9_pesudo_highscore_2
# python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_4_w9_pesudo_highscore_2_40000.pth
# config=configs.default_4_w9_pesudo_highscore_2
# python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

zip default_4_pesudo_highscore_2_continue_7_32000.zip ../exp/20200316_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_7_32000/*
mv -f default_4_pesudo_highscore_2_continue_7_32000.zip ../exp
# zip result_32000_4_pesudo_highscore_2.zip ../exp/20200311_hrnet_w48_ocr_up4_default_4_w9_pesudo_highscore_2_32000/*
# mv -f result_32000_4_pesudo_highscore_2.zip ../exp
# zip result_40000_4_pesudo_highscore_2.zip ../exp/20200311_hrnet_w48_ocr_up4_default_4_w9_pesudo_highscore_2_40000/*
# mv -f result_40000_4_pesudo_highscore_2.zip ../exp