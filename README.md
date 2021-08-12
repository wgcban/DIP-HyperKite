# DIP-HyperKite: Hyperspectral Pansharpening Based on Improved Deep Image Prior and  Residual Reconstruction
[Wele Gedara Chaminda Bandara](https://www.researchgate.net/profile/Chaminda-Bandara-4), Jeya Maria Jose Valanarasu (https://jeya-maria-jose.github.io/research/), [Vishal M. Patel](http://www.rci.rutgers.edu/~vmp93/)

[[Paper Link](https://arxiv.org/abs/2107.02630)] (Preprint submitted to IEEE Transactions on Geoscience and Remote Sensing)

In this paper, we have presented a novel approach for HS pansharpening, which mainly consists of three steps: (1) Up-sampling the LR-HSI via DIP, (2) Predicting the residual image via over-complete HyperKite, and (3) Obtaining the final fused HSI by summation. The previously proposed DIP methods for HS up-sampling only impose a constraint in the spectral-domain  by utilizing LR-HSI. To better preserve both spatial and spectral information, we first exploited an additional spatial constraint to DIP by utilizing the available PAN image, thereby introduced both spatial and spectral constraints to the DIP. The comprehensive experiments conducted on three HS datasets showed that our proposed spatial+spectral loss function significantly improved the quality of up-sampled HSIs in CC, RMSE, RSNR, SAM, ERGAS, and PSNR performance measures. Next, in the residual prediction task, we have shown that the residual component between  up-sampled HSI and the reference HSI primarily consists of edge information and very fine structures. Motivated by this observation, we proposed a novel over-complete deep-learning network for the residual prediction task. In contrast to the conventional under-complete representations, we have shown that our over-complete network is competent to focus on high-level features such as edges and fine structures by constraining the receptive field of the network. Finally, the fused HSI is obtained by adding the residual HSI and the up-sampled HSI. The comprehensive experiments conducted on three HS datasets demonstrated the superiority of our DIP-HyperKite over the other state-of-the-art results in terms of qualitative and quantitative evaluations.

## Citation
    @inproceedings{bandara2021hyperspectral,
    title={Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction},
    author={Bandara, Wele Gedara Chaminda and Valanarasu, Jeya Maria Jose and Patel, Vishal M},
    journal={arXiv preprint arXiv:2107.02630},
    year={2021}
    } 

<p align="center">
<img src="sample_results/121_input.jpg" width="300px" height="200px"/>         <img src="sample_results/121_our.jpg" width="300px" height="200px"/>
<img src="sample_results/38_input.jpg" width="300px" height="200px"/>         <img src="sample_results/38_our.jpg" width="300px" height="200px"/>
</p>