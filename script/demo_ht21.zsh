#!/usr/bin/env zsh

ANNO_PATH="datasets/mot/annotations"
HT21_PATH="datasets/mot/HT21"

demo_cmd="python tools/demo_track.py"
cfg_file="-f exps/ht21/yolox_x_ht21.py"

## Arguments defined in tools/demo_track.py
  # `--fps 25` means setting the frame rate of the input video, which also determines the number of frames for tentative tracks.
  # `--match_thresh 0.8` means reject a matching if IoU < 0.2
  # `--track_thresh 0.5` means initialize an unmatched detection which score > 0.6 as a new tracklet after the association is done.
  # `--min_box_area 0` (default: 10) means filter out detections which area < 0 in the current frame, after tracking is done.
  # `--max_aspect_ratio 10` (default: 1.6) means filter out detections which box_w/box_h >= 10 in the current frame, after tracking is done.
  # `--mot20` means do not fuse detection score into IoU matching distance in the first and third association.
  # `--with_deduplication` means do not remove duplicate tracks between tracked tracks and lost tracks
  # `--save_result` means saving tracking results in MOT's standard format into a txt file for evaluation and saving images with rendered results for visualization.
  # `--save_video` means generating a video from images with rendered detection & tracking results.
params="$cfg_file --fps 25 --det_thresh_lo 0.1 --det_thresh_hi 0.6 --match_thresh 0.8 --track_thresh 0.6"
params+=" --min_box_area 0 --max_aspect_ratio 10"
## Baseline: no score fusion and no deduplication
params+=" --mot20 --with_deduplication"
## Ablation-1: add deduplication
# params+=" --mot20"
## Ablation-2: fusing detection score into IoU distance before the matching in the first and third association
# params+=" --with_deduplication"
# params+="--save_result --save_video"
# params+=" --save_video"
params+=" --save_result"


def infer_on_pre_detections() {
    split=$1  # "train" or "test"
    det_json="$ANNO_PATH/ht21_$split""_public_det.json"
    # det_json="$HT21_PATH/det_yolov5/yolov5_on_ht21_$split.json"

    cmd="$demo_cmd ht21-$split $params --det_json $det_json"
    echo $cmd
    eval $cmd
    echo "\n"
}


arg1=$1
if [[ $arg1 == "predet" ]] {
    infer_on_pre_detections $2
}


## Scripts
# ex1: infer on HT21-train/test set with public/private detections
# cmd: ./script/demo_ht21.zsh predet train/test
