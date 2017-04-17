using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public interface ILayerFactory
    {
        ILayer CreateProduct(IKernelDescriptor descriptor);
    }
}
