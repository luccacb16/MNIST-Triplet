cmd="python3 train.py --num_images 59904 --batch_size 1024 --epochs 20 --margin 1 --num_workers 0 --data_path ./data/ --checkpoint_path ./checkpoints/"
echo $cmd
$cmd