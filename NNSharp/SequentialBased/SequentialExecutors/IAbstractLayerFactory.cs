using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using NNSharp.SequentialBased.SequentialLayers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialExecutors
{
    public interface IAbstractLayerFactory
    {
        ILayer CreateProduct(IKernelDescriptor descriptor);
    }
}
