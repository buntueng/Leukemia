#!/bin/bash

# Stop the script if any command fails
set -e

echo "==============================================="
echo "   Starting Leukemia Classification Training Pipeline"
echo "==============================================="

# 1. EfficientNetB3
if [ -f "efficientnetb3.py" ]; then
    echo -e "\n>>> [1/6] Running EfficientNetB3..."
    python3 efficientnetb3.py
else
    echo -e "\n[SKIP] efficientnetb3.py not found."
fi

# 2. InceptionV3
if [ -f "inceptionv3.py" ]; then
    echo -e "\n>>> [2/6] Running InceptionV3..."
    python3 inceptionv3.py
else
    echo -e "\n[SKIP] inceptionv3.py not found."
fi

# 3. MobileNetV3
if [ -f "mobilenetv3.py" ]; then
    echo -e "\n>>> [3/6] Running MobileNetV3..."
    python3 mobilenetv3.py
else
    echo -e "\n[SKIP] mobilenetv3.py not found."
fi

# # 4. ResNet50
# if [ -f "resnet50.py" ]; then
#     echo -e "\n>>> [4/6] Running ResNet50..."
#     python3 resnet50.py
# else
#     echo -e "\n[SKIP] resnet50.py not found."
# fi

# # 5. VGG19
# if [ -f "vgg19.py" ]; then
#     echo -e "\n>>> [5/6] Running VGG19..."
#     python3 vgg19.py
# else
#     echo -e "\n[SKIP] vgg19.py not found."
# fi

# # 6. Xception
# if [ -f "xception.py" ]; then
#     echo -e "\n>>> [6/6] Running Xception..."
#     python3 xception.py
# else
#     echo -e "\n[SKIP] xception.py not found."
# fi

echo "==============================================="
echo "   All Training Jobs Completed Successfully!   "
echo "==============================================="