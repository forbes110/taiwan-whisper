import os
import os.path as osp
import argparse
import glob
import subprocess
import tqdm

SAMPLE_RATE = 16000

def validate_data_dir(data_dir):
    video_fpaths = glob.glob(osp.join(data_dir, 'raw', "*.mp4")) + glob.glob(osp.join(data_dir, 'raw', "*.mov"))
    srt_failed_list = []
    srt_fpaths = []
    for video_fpath in video_fpaths:
        # print(video_fpath)
        vid = video_fpath.split('/')[-1].split('.')[0]
        revised_srt_fpath = osp.join(data_dir, 'raw/revised', f"{vid}.srt")
        if not osp.exists(revised_srt_fpath):
            srt_failed_list.append(revised_srt_fpath)
        else:
            srt_fpaths.append(revised_srt_fpath)
        # v2a_cmd = f"ffmpeg -hide_banner -loglevel error -y -i {video_fpath} -ac 1 -ar {SAMPLE_RATE} {audio_fpath}"
        # # v2a_cmd = f"ffmpeg -y -i {video_fpath} -ac 1 -ar {SAMPLE_RATE} {audio_fpath}" # enable logs
        # subprocess.run(v2a_cmd.split())
    if len(srt_failed_list) > 0:
        print(srt_failed_list)
        raise ValueError("Some video files do not have their matched srt files.")
    else:
        print("All the video files have their matched srt files.")
    return video_fpaths, srt_fpaths

def formulate_data(data_dir, video_fpaths, srt_fpaths):
    def _parse_id(id):
        id = id.replace(' ', '-')
        id = id.replace('.', '_')
        id = id.replace('_projector-blackboard', '')
        id = id.replace('_projector', '')
        id = id.replace('_Movie', '')
        return id

    for video_fpath, srt_fpath in zip(video_fpaths, srt_fpaths):
        v_format = '.mp4' if video_fpath.endswith('.mp4') else '.mov'
        id = video_fpath.split('/')[-1].split(v_format)[0]
        new_id = _parse_id(id)
        if new_id != id:
            print(f"Renaming {id} to {new_id}")
        new_video_fpath = osp.join(data_dir, 'raw', f"{new_id}.mp4")
        new_srt_fpath = osp.join(data_dir, 'raw/revised', f"{new_id}.srt")
        os.rename(video_fpath, new_video_fpath)
        os.rename(srt_fpath, new_srt_fpath)

def generate_audio(data_dir, video_fpaths):
    audio_dir = osp.join(data_dir, 'raw/audio')
    if not osp.exists(audio_dir):
        os.makedirs(audio_dir)
    for video_fpath in tqdm.tqdm(video_fpaths, total=len(video_fpaths)):
        v_format = '.mp4' if video_fpath.endswith('.mp4') else '.mov'
        id = video_fpath.split('/')[-1].split(v_format)[0]
        audio_fpath = osp.join(audio_dir , f"{id}.wav")
        v2a_cmd = f"ffmpeg -hide_banner -loglevel error -y -i {video_fpath} -ac 1 -ar {SAMPLE_RATE} {audio_fpath}"
        # v2a_cmd = f"ffmpeg -y -i {video_fpath} -ac 1 -ar {SAMPLE_RATE} {audio_fpath}" # enable logs
        subprocess.run(v2a_cmd.split())

def main(args):
    print(args)
    data_dir = args.data_dir
    video_fpaths, srt_fpaths = validate_data_dir(data_dir)
    print(len(video_fpaths), len(srt_fpaths))
    assert len(video_fpaths) == len(srt_fpaths)
    # formulate_data(data_dir, video_fpaths, srt_fpaths)
    # generate_audio(data_dir, video_fpaths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        default="",
        help="data directory",
    )
    args = parser.parse_args()

    main(args)