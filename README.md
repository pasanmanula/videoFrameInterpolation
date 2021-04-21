# Video Frame Interpolation
Video Frame Interpolation

Capturing video(s) at a higher frame rate is challenging, and yet existing equipment is costly. One
of the major drawbacks when increasing the frame rate of a digital camera is that the video
resolution will drop significantly. The main reason is, inbuilt video processors are struggling to
process and store high data rates. Therefore, if there is a post-processing method to generate
intermediate video frames after capturing, it will reduce the complexity and the cost of video
processors.

Video frame interpolation can be used for different computer vision applications, including, but
not limited to, slow motion generation [1], frame rate up-conversion [2], and frame recovery in
video streaming (video compression). The videos with a high frame rate can avoid common
artifacts, such as motion blurriness and jittering, and therefore are visually more appealing to the
viewers. However, with the deep convolutional neural networks on video frame interpolation, it
is still challenging to generate high-quality frames due to large motion.

The following papers describe the recent developments in the video frame interpolation.
"Learning image matching by simply watching video, [3]", "Video frame synthesis using deep
voxel flow. [4]", and which explains how to train a model to synthesize in-between frames
directly. However "depth-aware video frame interpolation. [5]", was the selected model for this
project because it performs state-of-the art interpolation results.

[1] D. V. J. M. Y. E. L. M. a. J. K. H.Jiang, "High Quality Estimation of Multiple Intermediate
Frames for Video Interpolation," 2018.
[2] M. S. U.S. Kim, "New frame rate up-conversion algorithms with low computational
complexity," 2014.
[3] L. K. J. A. H. X. Z. a. Q. Y. G. Long, "Learning image matching by simply watching video,"
2016.
[4] R. Y. X. T. Y. a. A. A. Z. Liu, "Video frame synthesis using deep voxel flow," 2017.
[5] W.-S. L. C. M. X. Z. Z. G. M.-H. Y. Wenbo Bao, "Depth-Aware Video Frame Interpolation,"
2019. 

Original model:
@inproceedings{DAIN,
    author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan}, 
    title     = {Depth-Aware Video Frame Interpolation}, 
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year      = {2019}
}
