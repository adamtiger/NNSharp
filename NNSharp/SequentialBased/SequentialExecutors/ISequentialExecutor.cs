using NNSharp.Kernels;
using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;

namespace NNSharp.SequentialBased.SequentialExecutors
{
    public interface ISequentialExecutor
    {
        void Compile(List<IKernelDescriptor> descriptors);
        IData Execute(IData input);
        void SetWeights(List<IData> weights);
    }
}
