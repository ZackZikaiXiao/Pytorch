## 1.This web is about cross entropy function:  
[easy to understand](https://blog.csdn.net/qq_22210253/article/details/85229988)  

[mathematical](https://www.cnblogs.com/marsggbo/p/10401215.html)

## 2.three concepts in DP:
- iteration：表示1次迭代（也叫training step），每次迭代更新1次网络结构的参数；
- batch-size：1次迭代所使用的样本量；
- epoch：1个epoch表示过了1遍训练集中的所有样本。
> 例如:定义10000次迭代为1个epoch，若每次迭代的batch-size设为256，那么1个epoch相当于过了2560000个训练样本。(1个epoch有10000次迭代，一次迭代需要256个样本)
