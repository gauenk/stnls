# Dataset List

Finding the correct links to the datasets named in the references is a bit confusing. While the links I provide may perish in a just a few short months, I have aggregated them into one place in an attempt to consolidate the search for a brief moment in time.

Denoising
- SIDD [webpage](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php). The reported testing data is the benchmark dataset. The sRGB dataset (rather than RAW) is the version which is commonly reported. The validation dataset is about 1GB for all four files, or two pairs of (noisy,gt) images. The benchmark dataset is similar. To format the data with Python, I recommend generating their submission file (a "SubmitSrgb.mat" file) with their MATLAB code. Then read that file with `h5dfstorage` in Python and over-write the "DenoisedBlocksSrgb" with your own denoised video.
- DND [website](https://noise.visinf.tu-darmstadt.de/). You must register to use their dataset. The dataset is only 12GB.
- Starlight Denoising [website](https://github.com/monakhova/starlight_denoising/). This dataset is new, so there does not exist a standard. We choose to run denoising on the "curated, smaller dataset" of only 1.3GB, [linked here from their github](https://drive.google.com/drive/folders/1ztbuJElSdT2MTOm1RgGnSEDFXIsBHO5q).
- SMID [webpage](https://github.com/cchen156/Seeing-Motion-in-the-Dark). This dataset is huge (~128GB), and I feel the authors should invest in a smaller version so it is more usable for testing. So to get this dataset, free-up some hard drive space and buckle down for a wait. Since this dataset is not commonly tested on, I choose to use the pre-processed (and smaller) dataset which has VBM4D already run on it. The color looks funny, but that's part of the problem statement so it is downloaded correctly. The "long" is long exposure and the "short" is short exposure. The "M" folders contain the motion images for video testing. The other 202 folders contain static scenes.

Deblurring
- RealBlur [website](http://cg.postech.ac.kr/research/realblur/). Go to google drive and download RealBlur.tar.gz and RealBlur-Tele.tar.gz. I am not sure what the difference is yet.
- GoPro [website](https://seungjunnah.github.io/Datasets/gopro.html). The version for testing is either (I think) GOPRO_Large. I used the Google Drive link.
- HIDE [website](https://github.com/joanshen0508/HA_deblur). There is a link in the first paragraph of text.

Super-resolution: Each set has four folders: high-resolution (HR) and three downscaled, low-resolution (LR) images with some multiplier. I think they use BICUBIC downsampling but both PIL and cv2 don't give the same results as the downloaded low-resolution images. I have compared with all the downscaling methods in the package with no success, so I just use these LR images to a fair comparison.
- [Set5](https://huggingface.co/datasets/eugenesiow/Set5/tree/main/data)
- [Set14](https://huggingface.co/datasets/eugenesiow/Set14/tree/main/data)
- [BSD100](https://huggingface.co/datasets/eugenesiow/BSD100/tree/main/data)
- [Urban100](https://huggingface.co/datasets/eugenesiow/Urban100/tree/main/data)
