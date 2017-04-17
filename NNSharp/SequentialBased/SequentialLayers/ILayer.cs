using NNSharp.DataTypes;
using NNSharp.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public interface ILayer : IKernel
    {
        void SetInput(IData input);
        IData GetOutput();
        void SetWeights(IData weights);
    }
}
