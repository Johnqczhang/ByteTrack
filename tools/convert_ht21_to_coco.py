import os
import os.path as osp
import cv2
import numpy as np
import json
import copy

from functools import reduce


MOT_PATH = osp.join(osp.dirname(__file__), "../datasets/mot")
ANNO_PATH = osp.join(MOT_PATH, "annotations")
os.makedirs(ANNO_PATH, exist_ok=True)

HT21_CATEGORIES = {
    1: "pedestrian", 2: "static", 3: "ignored", 4: "person on vehicle"
}


def load_ht21_gt_txt(txt_path):
    """
    Load HT21 (CroHD) raw annotations from the gt.txt file for a video sequence.
    Each row in the txt file contains the following fields separated by commas.
        `frame_id, identity_id, box_x1, box_y1, box_w, box_h, ignored_flag, class_id, visible_flag`

    Returns:
        objects_per_frame (dict): {frame_id: list[anno1_dict, anno2_dict, ...]}
    """
    objects_per_frame = {}
    obj_ids_per_frame = {}  # To check that no frame contains two objects with the same id

    # count the number of both visible and occluded objects per category in the sequence.
    cat_cnt = {k: [0, 0] for k in HT21_CATEGORIES}
    print(f"Loading annotations from {txt_path}")

    annos = np.loadtxt(txt_path, dtype=np.float32, delimiter=',')
    for fields in annos:
        frame_id = int(fields[0])
        if frame_id not in objects_per_frame:
            objects_per_frame[frame_id] = []
        if frame_id not in obj_ids_per_frame:
            obj_ids_per_frame[frame_id] = set()

        obj_id = int(fields[1])
        assert obj_id != -1  # -1 indicates detected results
        assert obj_id not in obj_ids_per_frame[frame_id], (
            f"Error: multiple objects with the same id: {obj_id} in frame-{frame_id}."
        )
        obj_ids_per_frame[frame_id].add(obj_id)

        box = [float(x) for x in fields[2:6]]  # xywh
        assert box[2] > 0 and box[3] > 0
        area = box[2] * box[3]
        # filter out invalid object with empty box
        if area < 1:
            continue

        cat_id = int(fields[7])  # category id
        assert cat_id in HT21_CATEGORIES, f"Unknown object class id: {cat_id}"
        if int(fields[6]) == 0:  # ignored object
            cat_id = 3
        is_visible = int(fields[8])
        assert is_visible in [0, 1]
        cat_cnt[cat_id][is_visible] += 1

        iscrowd = 1 if (not is_visible or cat_id == 3) else 0

        objects_per_frame[frame_id].append({
            "category_id": cat_id,
            "obj_id": obj_id,
            "bbox": box,
            "area": area,
            "iscrowd": iscrowd,
        })

    # print statistics information
    cat_cnt = {HT21_CATEGORIES[k]: v for k, v in cat_cnt.items() if sum(v) > 0}
    num_frames = len(objects_per_frame)
    num_objs = len(reduce(lambda x, y : x.union(y), obj_ids_per_frame.values()))
    print(f"frames: {num_frames}, [invisible, visible] annos: {cat_cnt}, objs: {num_objs}\n")

    return objects_per_frame


def load_ht21_det_txt(txt_path):
    """
    Load HT21 (CroHD) public detections from the det.txt file for a video sequence.
    Each row in the txt file contains the following fields separated by commas.
        `frame_id, -1, box_x1, box_y1, box_w, box_h, confidence socre (%), class_id, visible_flag`

    Returns:
        dets_per_frame (dict): {frame_id: list[det1_dict, det2_dict, ...]}
    """
    dets_per_frame = {}
    print(f"Loading detections from {txt_path}")
    dets = np.loadtxt(txt_path, dtype=np.float32, delimiter=',')

    for fields in dets:
        frame_id = int(fields[0])
        if frame_id not in dets_per_frame:
            dets_per_frame[frame_id] = []

        box = [float(x) for x in fields[2:6]]  # xywh
        score = float(fields[6]) / 100  # confidence score in [0, 1]

        dets_per_frame[frame_id].append({
            "category_id": 1,
            "obj_id": -1,
            "bbox": box,
            "score": score,
        })

    num_dets = sum([len(v) for v in dets_per_frame.values()])
    print(f"{num_dets} detections loaded.\n")

    return dets_per_frame


def ht21_to_coco_json(args):
    json_dict = {
        "info": {"description": args["dataset_name"]},
        "categories": args["categories"],
        "images": [],
        "annotations": [],
        "detections": [],
    }

    # count the number of frames in the dataset
    imgs_cnt = {"valid": 0, "total": 0}
    categories = {cat["id"]: cat["name"] for cat in args["categories"]}
    categories[-1] = "ignored"
    # count the number of annotations per category in the dataset
    annos_cnt = {c: 0 for c in categories}
    # count the number of object indentities per category in the dataset
    objs_dict = {c: {} for c in categories}
    # count the width, height, aspect ratio, and area of all boxes per category in the dataset
    boxes_cnt = {c: [] for c in categories}
    ann_id = 0  # annotation id in the entire dataset, starting from 1
    is_train = args["is_train"]
    base_dir = "HT21-train" if is_train else "HT21-test"

    print(f"Starting COCOfying {args['dataset_name']} - {'train' if is_train else 'test'} set ...")

    for seq_name, imgs_dir in zip(args["seqs_name"], args["imgs_dir"]):
        seq_id = int(seq_name[-2:])
        # dict, {frame_id: [anno1_dict, anno2_dict, ...]}
        seq_annos = args["seqs_annos"][seq_name]
        # dict, {frame_id: [det1_dict, det2_dict, ...]}
        seq_dets = args["seqs_dets"][seq_name]

        img_files = sorted([
            img for img in os.listdir(imgs_dir) if img.endswith(args["imExt"])
        ])
        num_imgs = len(img_files)  # number of frames in the sequence
        img_cnt = 0  # count the number of frames with valid (non-empty) annotations
        # count the number of annotations per category in the sequence.
        seq_ann_cnt = {c: 0 for c in categories}
        # count the number of object identities per category in the sequence.
        seq_obj_cnt = {c: 0 for c in categories}
        # initialize as 0 to check the resolution across all frames
        img_h = img_w = 0

        for filename in img_files:
            # frame number shown in the filename of image file, starting from 1.
            frame_id = int(filename.split('.')[0])

            img_path = f"{base_dir}/{seq_name}/img1/{filename}"
            img = cv2.imread(osp.join(imgs_dir, filename))
            if img_h > 0 and img_w > 0:
                assert img.shape[0] == img_h and img.shape[1] == img_w
            img_h, img_w = img.shape[:2]

            # image number in the entire dataset, starting from 1.
            img_id = imgs_cnt["total"] + frame_id
            img_info = {
                "file_name": img_path,
                "id": img_id,
                "prev_image_id": img_id - 1 if frame_id > 1 else -1,
                "next_image_id": img_id + 1 if frame_id < num_imgs else -1,
                "height": img_h,
                "width": img_w,
                "frame_id": frame_id,
                "seq_id": seq_id,
                "num_frames": num_imgs
            }
            json_dict["images"].append(img_info)

            # add `img_id` to the field of each detection
            if frame_id in seq_dets:
                for det in seq_dets[frame_id]:
                    det.update({
                        "image_id": img_id,
                        "prev_image_id": img_id - 1 if frame_id > 1 else -1,
                        "next_image_id": img_id + 1 if frame_id < num_imgs else -1,
                        "seq_id": seq_id,
                        "frame_id": frame_id,
                        "num_frames": num_imgs,
                    })
                    json_dict["detections"].append(det)

            if not is_train:
                continue
            # skip frame with empty annotation
            if frame_id not in seq_annos or len(seq_annos[frame_id]) == 0:
                continue

            annos_cur_frame = seq_annos[frame_id]
            # count a frame which contains at least one valid instance
            if any(obj["iscrowd"] == 0 for obj in annos_cur_frame):
                img_cnt += 1

            for obj in annos_cur_frame:
                if obj["iscrowd"]:
                    # ignored object will not be considered in both training and evaluation
                    cat_id = -1
                elif obj["category_id"] != 1:
                    # static person and person on vehicle will be considered only in detection by setting `category_id = 1,
                    # and will not be considered in tracking by setting `obj_id = -1`
                    cat_id = 1
                    obj_id = -1
                else:
                    obj_id = obj["obj_id"]
                    cat_id = obj["category_id"]

                annos_cnt[cat_id] += 1
                seq_ann_cnt[cat_id] += 1
                if cat_id == -1:
                    continue

                if obj_id != -1:
                    # set a unique object id in the entire dataset: seqName_catId_objId
                    obj_uId = f"{seq_name}_{cat_id}_{obj_id}"
                    if obj_uId in objs_dict[cat_id]:  # an existed object
                        obj_id = objs_dict[cat_id][obj_uId]
                    else:  # a new object
                        obj_id = sum([len(v) for v in objs_dict.values()]) + 1  # starting from 1
                        objs_dict[cat_id][obj_uId] = obj_id
                        seq_obj_cnt[cat_id] += 1

                # count head bbox statistics of all valid person objects (pedestrain, static person and person on vehicle)
                box_w, box_h = obj["bbox"][2:4]
                boxes_cnt[cat_id].append([box_w, box_h, box_w / box_h, box_w * box_h])
                # hard copy of each object's annotation to avoid reference
                anno = copy.deepcopy(obj)
                # update and save the COCO-format annotation of an object
                anno.update(
                    image_id=img_id,
                    id=ann_id + 1,
                    obj_id=obj_id,
                    seq_id=seq_id,
                    category_id=cat_id,
                )
                json_dict["annotations"].append(anno)
                ann_id += 1

        imgs_cnt["total"] += num_imgs
        if is_train:
            imgs_cnt["valid"] += img_cnt
            seq_ann_cnt = {categories[c]: v for c, v in seq_ann_cnt.items() if v > 0}
            seq_obj_cnt = {categories[c]: v for c, v in seq_obj_cnt.items() if v > 0}
            print(
                f"seq: {seq_name}, frames: {img_cnt}/{num_imgs}, size: {img_w}x{img_h},",
                f"annos: {seq_ann_cnt}, objs: {seq_obj_cnt}"
            )
        else:
            print(f"seq: {seq_name}, frames: {num_imgs}, size: {img_w}x{img_h}")

    if is_train:
        annos_cnt = {categories[c]: v for c, v in annos_cnt.items() if v > 0}
        objs_cnt = {categories[c]: len(v) for c, v in objs_dict.items() if len(v) > 0}
        print(
            f"dataset: {args['dataset_name']} - train set\n"
            f"frames: {imgs_cnt['valid']}/{imgs_cnt['total']} (valid/total)\n"
            f"annos: {annos_cnt}, objs: {objs_cnt}"
        )
        for c in boxes_cnt:
            if len(boxes_cnt[c]) == 0:
                continue
            boxes = np.array(boxes_cnt[c])
            print(
                f"{categories[c]} (whas): "
                f"min: {boxes.min(axis=0)}, max: {boxes.max(axis=0)}, mean: {boxes.mean(axis=0)}"
            )
        print()
    else:
        print(f"dataset: {args['dataset_name']} - test set, total frame: {imgs_cnt['total']}\n")

    return json_dict


def gen_ht21(is_train=True):
    imgs_dir = osp.join(MOT_PATH, "ht21", "HT21-train" if is_train else "HT21-test")

    args = {}
    args["seqs_name"] = sorted([
        seq for seq in os.listdir(imgs_dir) if seq.startswith("HT21")
    ])
    # Load MOT-format annotations from txt file for train set
    args["seqs_annos"] = {
        seq: load_ht21_gt_txt(osp.join(imgs_dir, seq, "gt/gt.txt")) if is_train else []
        for seq in args["seqs_name"]
    }
    # Load public detections from txt file
    args["seqs_dets"] = {
        seq: load_ht21_det_txt(osp.join(imgs_dir, seq, "det/det.txt"))
        for seq in args["seqs_name"]
    }
    args["imgs_dir"] = [
        osp.join(imgs_dir, seq, "img1") for seq in args["seqs_name"]
    ]
    args["categories"] = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"},
        # {"id": cat_id, "name": name, "supercategory": "person"}
        # for cat_id, name in HT21_CATEGORIES.items()
    ]
    args["dataset_name"] = "Crowd of Heads Dataset (CroHD)"
    args["imExt"] = ".jpg"
    args["is_train"] = is_train

    json_data = ht21_to_coco_json(args)
    detections = json_data.pop("detections")
    json_file = osp.join(ANNO_PATH, f"ht21_det_{'train' if is_train else 'test'}.json")
    with open(json_file, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"COCO-format detection results saved in {json_file}.")

    if not is_train:
        json_file = osp.join(ANNO_PATH, "ht21_test.json")
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"COCO-format image info data saved in {json_file}.")
        return

    json_file = osp.join(ANNO_PATH, "ht21_train.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"COCO-format annotations saved in {json_file}.")

    """ Split train data into training and validation sets in a leaving-one-out fashion."""
    for seq_name in args["seqs_name"]:
        seq_id = int(seq_name[-2:])
        out = {k: v for k, v in json_data.items() if k not in ["images", "annotations"]}
        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] != seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] != seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_train_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-format annotations saved in {json_file}.")
        dets = [
            copy.deepcopy(det) for det in detections if det["seq_id"] != seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_det_train_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(dets, f, indent=4)
        print(f"COCO-format detection results saved in {json_file}.")

        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] == seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] == seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_val_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-format annotations saved in {json_file}.")
        dets = [
            copy.deepcopy(det) for det in detections if det["seq_id"] == seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_det_val_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(dets, f, indent=4)
        print(f"COCO-format detection results saved in {json_file}.")


if __name__ == "__main__":
    gen_ht21(is_train=True)
    print()
    gen_ht21(is_train=False)
