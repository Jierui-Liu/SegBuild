###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-03-03 08:41:39
 # @Description : 
 ###
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/LinHonghui/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/LinHonghui/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd tools
conda activate SegBulid

# config_file="../configs/server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_batch_12_step_size_up_5000.py"  
config_file="../configs/server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_batch_16_step_size_up_2500.py"
val_image_dir="/home/LinHonghui/Datasets/SegBulid/test_dataset_npy/image_bin/"
# path="/home/LinHonghui/Project/DrivenData_2020_SegBulid/output/model/hrnet_w48_up8_server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_batch_12_step_size_up_5000_40000.pth"
# path="/home/LinHonghui/Project/DrivenData_2020_SegBulid/output/model/hrnet_w48_up8_server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_batch_16_step_size_up_2500_30000.pth" # 78.60
# path="/home/LinHonghui/Project/DrivenData_2020_SegBulid/output/model/hrnet_w48_up8_server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_batch_16_step_size_up_2500_25000.pth" #7860
path="/home/LinHonghui/Project/DrivenData_2020_SegBulid/output/model/hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_p79_batch_12_step_size_up_5000_20000.pth"
python inference.py -device 0 1 2 3 -path $path -val_batch_size 24 -config_file $config_file \
    -val_image_dir $val_image_dir
