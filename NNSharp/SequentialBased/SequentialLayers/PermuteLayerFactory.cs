using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class PermuteLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Permute)
            {
                Permute permute = descriptor as Permute;

                ILayer layer = new PermuteLayer(permute.Dim1, permute.Dim2, permute.Dim3);

                return layer;
            }

            return null;
        }
    }
}
