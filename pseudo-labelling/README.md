1. Given a directory with lots of audios(e.g., *.flac, *.wav, *.m4a), make a tsv file "dataset_path" as first-step dataset metadata.
for example: a `sample.tsv` file with content:
`audio_path
/mnt/dataset_1T/tmp_dir/example/lcMAHaXJflI.flac
/mnt/dataset_1T/tmp_dir/example/p8J-WHSz47E.flac
/mnt/dataset_1T/tmp_dir/example/q4HmpXp0I-g.flac
/mnt/dataset_1T/tmp_dir/example/StRDn_NWGvQ.flac
/mnt/dataset_1T/tmp_dir/example/ZmpMTBnCI4w.flac
/mnt/dataset_1T/tmp_dir/example/BN1YP9VIB08.flac
/mnt/dataset_1T/tmp_dir/example/n2LwQd_rZ5g.flac
/mnt/dataset_1T/tmp_dir/example/QZGURfv1DDQ.flac
`
2. Run `python3 filter_data.py` to make sure all sample rates are 16000 Hz & in flac format.
3. Run `bash inference.sh` to get psuedo label with time stamps.
4. Run `bash prepare_dataset.sh` to get segments with 30 secs for all data. 
5. Run `bash generate_metadata.py`