cd tools

path_7818="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/exp/7818"
path_7894="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/exp/7894"
path_7919="/home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/exp/7919"

python ensemble.py -path $path_7818 $path_7894 $path_7919 \
                -save_dir /home/chenbangdong/cbd/LinHonghui/DrivenData_2020_SegBulid/SegBuild/exp/7818_7894_7919