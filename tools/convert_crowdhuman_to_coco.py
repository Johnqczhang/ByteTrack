import os
import os.path as osp
import cv2
import json
import tqdm


DATA_PATH = osp.join(osp.dirname(__file__), "../datasets/crowdhuman")
ANNO_PATH = osp.join(DATA_PATH, "annotations")
os.makedirs(ANNO_PATH, exist_ok=True)


def load_annos_from_odgt(fpath):
    """
    Load CrowdHuman raw annotations from the .odgt file for the train/val set.
    Each line in the annotation file is a dict with two keys for an image,
        "ID": a string corresponds to the image filename,
        "gtboxes": [obj1_dict, obj2_dict, ...]
    Each dict of an object in the above list contains the following fields:
    {
        "fbox": [72, 202, 163, 503],  # full body bounding box
        "tag": "person",
        "hbox": [171, 208, 62, 83],  # head bounding box
        "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}
        "extra": {"box_id": 0, "occ": 0},
        "vbox": [72, 202, 163, 398],  # visible body bounding box
    }

    Returns:
        records (list(dict)): A list of dicts, each of which corresponds to an image and contains two keys "ID", "gtboxes".
    """
    assert osp.exists(fpath), f"Error: {fpath} not found."
    print(f"Loading annotations from {fpath}")
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


def crowdhuman_to_coco_json():
    json_dict = {
        "info": {"description": "CrowdHuman dataset"},
        "categories": [{"id": 1, "name": "person"}],
    }

    ann_id = 0
    for split in ["train", "val"]:
        json_dict["images"] = []
        anns_fbox, anns_hbox = [], []
        anns_data = load_annos_from_odgt(osp.join(ANNO_PATH, f"annotation_{split}.odgt"))

        img_cnt = 0
        # count the number of bboxes across both valid and ignored per type
        ann_cnt = {k: [0, 0] for k in ["fbox", "hbox"]}
        print(f"Starting COCOfying CrowdHuman - {split} set ...")

        for ann_data in tqdm.tqdm(anns_data):
            img_cnt += 1
            img_path = f"Crowdhuman_{split}/Images/{ann_data['ID']}.jpg"
            img = cv2.imread(osp.join(DATA_PATH, img_path))
            img_h, img_w = img.shape[:2]

            img_info = {
                "file_name": img_path,
                "id": img_cnt,  # image number in the dataset, starting from 1.
                "height": img_h,
                "width": img_w
            }
            json_dict["images"].append(img_info)

            for obj in ann_data["gtboxes"]:
                ann_id += 1
                fbox, vbox, hbox = obj["fbox"], obj["vbox"], obj["hbox"]  # fmt: xywh

                iscrowd = 0
                if "extra" in obj and "ignore" in obj["extra"] and obj["extra"]["ignore"] == 1:
                    iscrowd = 1

                ann_cnt["fbox"][iscrowd] += 1
                ann_fbox= {
                    "id": ann_id,
                    "category_id": 1,
                    "image_id": img_cnt,
                    "bbox": fbox,
                    "bbox_vis": vbox,
                    "area": float(fbox[2] * fbox[3]),
                    "iscrowd": iscrowd,
                }
                anns_fbox.append(ann_fbox)

                if "head_attr" in obj:
                    head_attr = obj["head_attr"]
                    if len(head_attr) == 0 or ("ignore" in head_attr and head_attr["ignore"] == 1):
                        iscrowd = 1
                ann_cnt["hbox"][iscrowd] += 1
                ann_hbox = {k: v for k, v in ann_fbox.items() if not k.startswith("bbox")}
                ann_hbox["bbox"] = hbox
                ann_hbox["area"] = float(hbox[2] * hbox[3])
                ann_hbox["iscrowd"] = iscrowd
                anns_hbox.append(ann_hbox)

        print(
            f"CrowdHuman - {split} set, images: {img_cnt},",
            f"full boxes: {ann_cnt['fbox']}, head boxes: {ann_cnt['hbox']}"
        )
        for k, anns in zip(["", "_head"], [anns_fbox, anns_hbox]):
            json_file = osp.join(ANNO_PATH, f"crowdhuman{k}_{split}.json")
            json_dict["annotations"] = anns
            with open(json_file, 'w') as f:
                json.dump(json_dict, f, indent=4)
            print(f"COCO-format annotations saved in {json_file}.")


if __name__ == '__main__':
    crowdhuman_to_coco_json()
