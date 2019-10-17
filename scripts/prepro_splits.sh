#!/bin/bash
# Usage: `./prepro_splits.sh`

if [ $1 == "activity" ] ||  [ $1 == "ALL" ]; then
    echo "Preprocessing activity dialogs"
    python prepro.py -input_json_train visdial_category_splits/train/activity.json -input_json_val visdial_category_splits/val/activity.json -input_json_test visdial_1.0_test.json -image_root visdial_images -output_json visdial_params_activity.json -output_h5 visdial_data_activity.h5

fi
