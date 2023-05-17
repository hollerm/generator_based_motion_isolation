# Latent-space disentanglement with untrained generator networks for the isolation of different motion types in video data
This repository provides the source code for the paper "Latent-space disentanglement with untrained generator networks for the isolation of different motion types in video data" as cited below.

 
## Requirements
The code was written and tested with Python 3.10.10 under Linux. No dedicated installation is needed for the code, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* [numpy], tested with version 1.24.3
* [scipy], tested with version 1.10.1
* [matplotlib], testd with version 3.7.1
* [torch], tested with version 2.0.0
* [cv2], tested with version 4.7.0
* [PIL], tested with version 9.5.0
* [imageio], tested with version 2.28.1
* [sys]
* [copyreg]
* [types]
* [pickle]
* [random]
* [Ipython]
* [os]
* [itertools]
* [copy]

We recomment to use torch.cuda to get GPU support.

## Examples

* To run a quick demo example, call "python demo.py" in a terminal. This should compute results with phantom data and store it in the folder "demo_results". In particular, a video will be stored in "demo_results/videos" and images as used in the paper will be stored in "demo_results/images".

* To re-compute experiments of the paper, call "python compute_paper_results.py". By default, this reproduces only the experiments with the phantom data. Reproducing the experiments with real data requires the data files for the real data to be stored in "data". These files are available upon request from martin.holler@uni-graz.at. After having obtained the real data, make sure to uncomment the respective lines of code in "compute_paper_results.py".

* To create images and videos for the paper from the computed results of the previous step, run "python create_paper_results.py". Again, if you want to create results also for real-data experiments, make sure to uncomment the appropriate lines in the code.

## Precomputed videos

* Videos for all results presented in the paper are available in "videos". Please see "videos_info.txt" for information on those videos.

## Authors of the code

* **Abdullah Abdullah** abdullahuom1@yahoo.com
* **Martin Holler** martin.holler@uni-graz.at 
* **Malena Sabate Landman** malena.sabate.landman@emory.edu


AA is affiliated with the Department of Mathematics, The Chinese University of Hong Kong, Hong Kong, Hong Kong. MH is affiliated with the Institute of Mathematics and Scientific Computing, University of Graz, Graz, Austria. MSL is affiliated with the Department of Mathematics, Emory University, Atlanta, USA

This code is the result of joint work of the above authors with **K. Kunisch** on the paper mentioned below. KK is affiliated with the Institute of Mathematics and Scientific Computing, University of Graz, Graz, Austria.

## Publications
If you use this code, please cite the following associated publication.

* Abdullah, A., Holler, M., Kunisch, K., Landman, M.S. (2023). Latent-Space Disentanglement with Untrained Generator Networks for the Isolation of Different Motion Types in Video Data. In: Calatroni, L., Donatelli, M., Morigi, S., Prato, M., Santacesaria, M. (eds) Scale Space and Variational Methods in Computer Vision. SSVM 2023. Lecture Notes in Computer Science, vol 14009. Springer, Cham. https://doi.org/10.1007/978-3-031-31975-4_25

## License

The code in this project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
