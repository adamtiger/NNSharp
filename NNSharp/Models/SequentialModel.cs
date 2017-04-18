using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.SequentialBased.SequentialExecutors;
using NNSharp.KernelDescriptors;
using NNSharp.Kernels;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Models
{
    [Serializable()]
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
            // Compiling the model
            compiler.Compile(descriptors);
            compiled = compiler;

            // Saving the input dimension
            Input2D input = descriptors[0] as Input2D;
            dim = new Dimension(input.Height, input.Width, input.Channels, input.Batch);
        }

        public IData ExecuteNetwork(IData input)
        {
            return compiled.Execute(input);
        }

        public Dimension GetInputDimension()
        {
            return dim;
        }

        [field: NonSerialized()]
        private List<IKernelDescriptor> descriptors;
        private ISequentialExecutor compiled;
        private Dimension dim;

    }
}
