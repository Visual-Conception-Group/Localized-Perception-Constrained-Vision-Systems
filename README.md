# Localized-Perception-Constrained-Vision-Systems

Despite the rapid advancement in the field of image recognition, the processing of
high-resolution imagery remains a computational challenge. However, this processing is
pivotal for extracting detailed object insights in areas ranging from autonomous vehicle
navigation to medical imaging analyses. Our study introduces a framework aimed at mit-
igating these challenges by leveraging memory efficient patch based processing for high
resolution images. It incorporates a global context representation alongside local patch
information, enabling a comprehensive understanding of the image content. In contrast
to traditional training methods which are limited by memory constraints, our method
enables training of ultra high resolution images. We demonstrate the effectiveness of
our method through superior performance on 7 different benchmarks across classifica-
tion, object detection, and segmentation. Notably, the proposed method achieves strong
performance even on resource-constrained devices like Jetson Nano

## Experiments On
1. Image Classification
2. Object Detection
3. Image Segmentation

## Conclusion
In this work, we address the significant challenge of efficiently processing high-resolution
images for tasks like classification, object detection, and segmentation under stringent mem-
ory constraints. Our innovative framework integrates localized patch-based processing with
a global contextual understanding, enabling comprehensive image analysis under memory
constraints. This approach not only preserves the fine-grained details necessary for accu-
rate object detection and segmentation but also incorporates global semantics essential for
robust classification performance. Experiments across seven distinct benchmarks, demon-
strates that our method achieves competitive performance. We also demonstrate training on
resource-constrained devices, such as the Jetson Nano