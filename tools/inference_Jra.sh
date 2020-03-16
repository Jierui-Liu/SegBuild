path=../output/model/hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_9_16000.pth
config=configs.default_4_pesudo_highscore_2_continue_9
python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_8_27000.pth
# config=configs.default_4_pesudo_highscore_2_continue_8
# python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

# path=../output/model/hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_8_36000.pth
# config=configs.default_4_pesudo_highscore_2_continue_8
# python inference.py -config_file $config  -device 0 -path $path # -save_dir $save_dir #-multi-gpu

zip default_4_pesudo_highscore_2_continue_9_16000.zip ../exp/20200316_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_9_16000/*
mv -f default_4_pesudo_highscore_2_continue_9_16000.zip ../exp
# zip default_4_pesudo_highscore_2_continue_8_27000.zip ../exp/20200316_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_8_27000/*
# mv -f default_4_pesudo_highscore_2_continue_8_27000.zip ../exp
# zip default_4_pesudo_highscore_2_continue_8_36000.zip ../exp/20200316_hrnet_w48_ocr_up4_default_4_pesudo_highscore_2_continue_8_36000/*
# mv -f default_4_pesudo_highscore_2_continue_8_36000.zip ../exp