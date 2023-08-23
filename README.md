# Efficient_Newton_Steps_for_Sparse_Feed_Forward_Neural_Networks
For solving a system of n-nonlinear equations Newton-Raphson method is utilized in conjunction with sparse feed-forward neural
networks. A factorized Newton method have been developed to perform Newton step to reduce the computational cost and make it
efficient for large amount of data.

1) Data is generated randomly by defining a system of n-nonlinear equations.
2) Not just an arbitrary sparse, a special tridiagonal sparse structure is used to create the network.
3) Custom tridiagonal layers are created from scratch to build the network.
4) After generating the model, Jacobian matrices are calculated for each layer to perform the Newton step.
5) Thomas algorithm is used to invert the Jacobian matrices of each layer.
6) The run time is calculated for this operation in each layer and compared it with the traditional method
   of computing Newton step.
7) Results show the factorized newton method to compute the Newton step is cheaper than the standard Newton method.
