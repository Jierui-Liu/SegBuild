cd tools

path_1="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/output/model/hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_weight_2_3cross_entropy2d_train_1_agri_aerial_batch_12_step_size_up_5000_50000.pth"
path_2="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/output/model/hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_weight_2_3cross_entropy2d_train_1_agri_aerial_batch_12_step_size_up_5000_40000.pth"
path_3="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/output/model/hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_weight_2_3cross_entropy2d_train_1_agri_aerial_batch_12_step_size_up_5000_30000.pth"
val_image_dir="/home/chenbangdong/cbd/DrivenDATA/test_dataset_npy/image_bin/"
python snapshot_ensemble.py -path $path_1 $path_2 $path_3 -device 2 3 \
                -save_dir /home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/exp/snapshot_hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_weight_2_3cross_entropy2d_train_1_agri_aerial_batch_12_step_size_up_5000_30000_40000_50000 \
                -val_image_dir $val_image_dir