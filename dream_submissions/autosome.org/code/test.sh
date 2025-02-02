echo "-------Testing..."
python3 test.py --seed 42 --valid_batch_size 1024 --valid_workers 8 --seqsize 150 --temp .TEMPDIR --use_single_channel --singleton_definition integer --gpu 0 --ks 7 --blocks 256 128 128 64 64 64 64 --resize_factor 4 --se_reduction 4 --final_ch 18 --target test_sequences.txt --delimiter tab --output_format tsv --output results.txt --model model/model_80.pth
echo "Done."