using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.SequentialBased.SequentialExecutors;
using NNSharp.KernelDescriptors;
using NNSharp.Kernels;
using NNSharp.DataTypes;

namespace NNSharp.Models
{
    public class SequentialModel
    {
        public SequentialModel()
        {
            descriptors = new List<IKernelDescriptor>();
        }

        public void Add(IKernelDescriptor descriptor)
        {
            descriptors.Add(descriptor);
        }

        public void SetWeights(List<IData> weights)
        {
            compiled.SetWeights(weights);
        }

        public void Compile(ISequentialExecutor compiler)
        {
            compiler.Compile(descriptors);
            compiled = compiler;
        }

        public IData ExecuteNetwork(IData input)
        {
            return compiled.Execute(input);
        }


        private List<IKernelDescriptor> descriptors;
        private ISequentialExecutor compiled;

    }
}
