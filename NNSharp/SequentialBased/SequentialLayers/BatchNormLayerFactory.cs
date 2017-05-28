using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class BatchNormLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is BatchNormalization)
            {
                BatchNormalization bnm = descriptor as BatchNormalization;

                ILayer layer = new BatchNormLayer(bnm.Epsilon);

                return layer;
            }

            return null;
        }
    }
}
