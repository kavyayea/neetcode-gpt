import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi=np.max(z)
        sum=0
        for i in range(len(z)):
            z_bar=z[i]-maxi;
            sum+=np.exp(z_bar)
        for i in range(len(z)):
            z_bar=z[i]-maxi;
            softmaxx=np.exp(z_bar)/sum
            z[i]=np.round(softmaxx,4)
        return z
