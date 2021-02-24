
# PPMM

Python3 implementation of the paper [Large-scale optimal transport map estimation using projection pursuit] (NeurIPS 2019)

Projection Pursuit Monge map (PPMM) is one of the projection-based empirical optimal transport map (OTM) estimation methods, which also includes Radon transformation method and Sliced method. Different from these methods, PPMM uses sufficient dimension reduction techniques to estimate the most “informative” projection direction in each iteration, resulting in a fast convergence rate in practice.

Feel free to ask if any question.

If you use this toolbox in your research and find it useful, please cite PPMM using the following bibtex reference:

```
@incollection{meng2019ppmm,
title = {Large-scale optimal transport map estimation using projection pursuit},
author = {{Meng}, Cheng and {Ke}, Yuan and {Zhang}, Jingyi and
 {Zhang}, Mengrui and {Zhong}, Wenxuan and {Ma}, Ping},
booktitle = {Advances in Neural Information Processing Systems 32},
year = {2019}
}
```

### Prerequisites
* Python (>= 3.6)
* Numpy (>= 1.11)
* Matplotlib (>= 1.5)
* For Optimal transport [Python Optimal Transport](https://pot.readthedocs.io/en/stable/) POT (>=0.5.1)


### What is included ?

* PPMM with sliced average variance estimator (SAVE) and directional regression (DR)

* Radon projection method, also called random projection method.

* Sliced methods, which is widely applied in Sliced Wasserstein distance calculation

* Empirical performance and runtimes comparaison with empirical Sinkhorn distance of [POT](https://github.com/rflamary/POT):

* Demo notebooks:
	- [ppmm_example.ipynb](ppmm_example.ipynb): Compare PPMM with other methods in calculating the Wasserstein distance
	- [color transfer.ipynb](color%20transfer.ipynb): Color Transfer using PPMM
	![image](https://github.com/ChengzijunAixiaoli/PPMM/blob/master/transfer1.png)


### Authors

* [Cheng Meng](https://github.com/ChengzijunAixiaoli)
* [Jingyi Zhang](https://github.com/joyeecat)
* [Mengrui Zhang](https://github.com/zhanzmr)
* [Tao Li](https://github.com/sherlockLitao)




## References

[1] Flamary Rémi and Courty Nicolas [POT Python Optimal Transport library](https://github.com/rflamary/POT)
